"""
AI-AI baseline simulation.

Generate simulated conversations for null hypothesis testing using multiple LLM providers.
Supports OpenAI, Anthropic, and HuggingFace (OpenAI-compatible).

This simulation matches the actual PENPAL human-AI experiment configuration:
- Temperature: 1.0
- Max tokens: 35
- System prompt: Collaborative storytelling prompt with pacing info

Context management per provider (matching experiment server/adapters):
- OpenAI: Native conversation threading via Conversations API.
          Only current turn text sent as 'input'; provider manages history.
          System prompt sent as 'instructions'.
          (Matches server/adapters/OpenAIAdapter.ts)
- Anthropic: Routed through an OpenAI-compatible chat/completions endpoint
             (e.g. OpenRouter), the SAME transport the experiment uses for Claude.
             System prompt is included in the messages array as {role: 'system'}
             via buildMessagesForCompat, NOT sent as Anthropic's native 'system'
             parameter. Requires a base_url.
             (Matches server/adapters/ClaudeAdapter.ts + server/utils/messages.ts)
- HuggingFace: Stateless. Builds [system, ...history, user: current_input] via
               buildMessagesForCompat. Messages trimmed to 12k char limit.
               (Matches server/adapters/VLLMAdapterA.ts + server/utils/messages.ts)

Post-processing (matching experiment frontend src/services/aiService.jsx):
- After every generation, if the response begins with the partner's input
  (case-insensitive), strip the echo. Applied uniformly to all providers.

Symmetric word-truncation design (matching the human-human experiment):
- BOTH agents have their output word-truncated by random.randint(2, 5)
  words after each generation. The truncated text is permanent — both
  stored AND passed to the next agent as its user input.
- Mirrors truncateUserInput from src/utils/submissionHelpers.js, applied
  to both partners (analogous to the human-human condition where both
  humans' inputs were truncated). Deployed JS uses a session-constant
  Math.floor(Math.random()*5)+1 in USER_INPUT_LIMITS.TRUNCATE_WORDS;
  the simulation re-randomizes per turn for cleaner statistical properties.
- Both agents use max_tokens=40, roughly matching the experiment's human
  character limit (USER_INPUT_LIMITS.MIN_CHARS=130 / MAX_CHARS=190 in
  src/constants/bookConstants.js). Word-truncation then trims a few words
  off the end so each side's input looks similarly mid-thought.
- Rationale for symmetric (not asymmetric) in AI-AI: in HH and AI-AI both
  partners are equally constrained, so symmetric treatment matches HH and
  avoids introducing a within-condition role nuisance variable. In HA the
  asymmetry is intrinsic (human vs AI), so truncation there is asymmetric
  by necessity.
"""

from typing import Optional, List, Dict, Literal
from abc import ABC, abstractmethod
import pandas as pd
import os
import time
import random
from tqdm import tqdm


# Provider type definitions
ProviderType = Literal["openai", "anthropic", "huggingface"]


def trim_messages_to_char_limit(messages: List[Dict[str, str]], limit: int = 12000) -> List[Dict[str, str]]:
    """Trim messages to fit within character limit, keeping most recent messages.
    
    Matches the experiment's trimMessagesToCharLimit utility (server/utils/messages.ts).
    Keeps system message (if first) plus as many recent messages as fit within limit.
    Trims from the oldest first, always preserving the most recent messages.
    """
    if not messages:
        return messages
    
    system_message = messages[0] if messages[0].get('role') == 'system' else None
    rest = messages[1:] if system_message else messages[:]
    
    remaining = limit
    kept: List[Dict[str, str]] = []
    
    for msg in reversed(rest):
        content_length = len(msg.get('content', ''))
        if content_length > remaining and kept:
            continue
        remaining -= content_length
        kept.insert(0, msg)
        if remaining <= 0:
            break
    
    if system_message:
        kept.insert(0, system_message)

    return kept


def build_messages_for_compat(
    system_prompt: Optional[str],
    history: Optional[List[Dict[str, str]]],
    user_input: str,
    char_limit: int = 12000,
) -> List[Dict[str, str]]:
    """Mirror of server/utils/messages.ts buildMessagesForCompat.

    Prepends a system message (unless history already starts with one), copies
    history skipping empty content, appends the current user input if non-empty,
    then trims to the char limit.
    """
    history_copy: List[Dict[str, str]] = [dict(m) for m in (history or [])]
    result: List[Dict[str, str]] = []

    if system_prompt:
        first = history_copy[0] if history_copy else None
        if not first or first.get("role") != "system":
            result.append({"role": "system", "content": system_prompt})

    for msg in history_copy:
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        result.append(msg)

    if user_input and user_input.strip():
        result.append({"role": "user", "content": user_input})

    return trim_messages_to_char_limit(result, limit=char_limit)


def truncate_user_input(text: str, words_to_truncate: int = 2) -> str:
    """Mirror of truncateUserInput from src/utils/submissionHelpers.js.

    Removes the last `words_to_truncate` whitespace-separated words from
    `text`. If the text has fewer words than that, returns an empty string
    (matching the JS behavior of `slice(0, len - n)` when n >= len).
    """
    if not isinstance(text, str):
        return ""
    trimmed = text.strip()
    if not trimmed:
        return ""
    words = trimmed.split()  # \\s+ splits, matches JS split(/\\s+/)
    if words_to_truncate <= 0:
        return trimmed
    clipped = words[: max(0, len(words) - words_to_truncate)]
    return " ".join(clipped)


def strip_input_echo(response_text: str, user_input: str) -> str:
    """Mirror of the echo-strip in src/services/aiService.jsx generateAIResponse.

    If the response begins (case-insensitively) with the partner's input,
    strip that prefix. Applied to all providers in the experiment frontend.
    """
    ai = (response_text or "").strip()
    ui = (user_input or "").strip()
    if ai and ui and ai.lower().startswith(ui.lower()):
        ai = ai[len(ui):].strip()
    return ai


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    Interface matches the experiment's ModelAdapter pattern:
    - system_prompt: Updated each turn with pacing metadata
    - user_input: Just the current turn's text from the partner
    - history: Prior turns as alternating user/assistant messages
    """
    
    @abstractmethod
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate a response from the model.
        
        Args:
            system_prompt: System instructions with pacing metadata
            user_input: Current turn's text from the partner (not cumulative story)
            history: Prior conversation as alternating user/assistant messages.
                     Ignored by threaded providers (OpenAI). Required by
                     stateless providers (Anthropic, HuggingFace).
            temperature: Sampling temperature (default 1.0 to match experiment)
            max_tokens: Maximum output tokens (default 35 to match experiment)
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def reset_conversation(self):
        """Reset conversation state for a new story."""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with native conversation threading.
    
    Matches experiment's OpenAIAdapter (server/adapters/OpenAIAdapter.ts):
    - Uses responses.create() with conversation threading
    - Sends system_prompt as 'instructions' (not seeded in conversation items)
    - Sends user_input as 'input' (just current turn text, not cumulative story)
    - Provider manages full history internally via conversation ID
    - history parameter is ignored (provider handles it)
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.conversation_id = None
    
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate using OpenAI's responses API with conversation threading.
        
        Matches experiment's OpenAIAdapter.respond():
        - Creates conversation with empty items on first call
        - Sends only current user_input as 'input' (not cumulative story)
        - Uses 'instructions' for system prompt (updated each turn)
        - Conversation thread manages all history internally
        - history parameter is ignored
        """
        if self.conversation_id is None:
            # Match experiment: create conversation with empty items
            # (experiment: this.client.conversations.create({ metadata: ..., items: [] }))
            conversation = self.client.conversations.create(
                metadata={"topic": "ai-ai-simulation"},
                items=[]
            )
            self.conversation_id = conversation.id
        
        # Match experiment: send instructions + input only, let thread handle history
        # (experiment: this.client.responses.create({ instructions: systemPrompt, input: userInput, conversation: id }))
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conversation_id,
            instructions=system_prompt,
            input=user_input,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Extract text from response (try output_text first, matching experiment)
        out = ""
        if hasattr(response, 'output_text') and response.output_text:
            out = response.output_text
        elif hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    if isinstance(item.content, str):
                        out += item.content
                    elif isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                out += content_item.text
        
        return out.strip()
    
    def reset_conversation(self):
        """Reset conversation ID for new story."""
        self.conversation_id = None


class AnthropicProvider(BaseProvider):
    """Claude provider via an OpenAI-compatible endpoint.

    Matches experiment's ClaudeAdapter (server/adapters/ClaudeAdapter.ts), which
    routes Claude through an OpenAI-compatible chat/completions endpoint
    (e.g. OpenRouter) — NOT Anthropic's native messages API.

    - STATELESS: no internal history
    - System prompt is the first message in the messages array
      ({role: 'system', ...}), placed by buildMessagesForCompat
    - Messages trimmed to 12k char limit
    - Requires base_url pointing at the compat endpoint root (e.g.
      "https://openrouter.ai/api/v1"). The OpenAI SDK appends /chat/completions.
    - model_name must be the proxy's identifier for Claude
      (e.g. "anthropic/claude-sonnet-4.5" on OpenRouter).
    """

    def __init__(self, api_key: str, model_name: str, base_url: str):
        from openai import OpenAI
        if not base_url:
            raise ValueError(
                "AnthropicProvider requires base_url (OpenAI-compatible endpoint). "
                "Set 'base_url' in model_configs to match the experiment's MODEL2_BASE_URL."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate via OpenAI-compatible chat completions.

        Mirrors ClaudeAdapter.respond(): builds messages with
        buildMessagesForCompat (system in-array), then POSTs to the compat
        endpoint with the same payload shape (model, messages, temperature,
        max_tokens).
        """
        messages = build_messages_for_compat(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            char_limit=12000,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""

    def reset_conversation(self):
        """No internal state to reset (stateless provider)."""
        pass


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider (OpenAI-compatible).
    
    Matches experiment's VLLM adapters (server/adapters/VLLMAdapterA.ts):
    - STATELESS: no internal conversation state
    - Builds [system, ...history, user: current_input] message array
    - Uses OpenAI-compatible chat completions endpoint
    - Messages trimmed to 12k char limit (matching server/utils/messages.ts)
    """
    
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://router.huggingface.co/v1"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
    
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate using HuggingFace's OpenAI-compatible API (stateless).
        
        Matches experiment's VLLM adapters via buildMessagesForCompat():
        - Builds [system, ...history, user: current_input]
        - No internal conversation state
        - Trims to 12k char limit
        """
        messages = build_messages_for_compat(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            char_limit=12000,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""
    
    def reset_conversation(self):
        """No internal state to reset (stateless provider)."""
        pass


def get_provider(provider_type: ProviderType, api_key: str, model_name: str, 
                 base_url: str = None) -> BaseProvider:
    """Factory function to create provider instances.
    
    Args:
        provider_type: One of 'openai', 'anthropic', 'huggingface'
        api_key: API key for the provider
        model_name: Model name/identifier
        base_url: Base URL for HuggingFace (OpenAI-compatible endpoint)
        
    Returns:
        Provider instance
    """
    if provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model_name=model_name, base_url=base_url)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(api_key=api_key, model_name=model_name, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


class SystemPrompts:
    """System prompts matching the actual PENPAL human-AI experiment.
    
    IMPORTANT: Both agents get the IDENTICAL prompt - the collaborative storytelling
    instruction. The only difference is turn order. This matches the actual experiment
    where both human and AI see the same instructions.
    """
    
    # Base prompt from the actual experiment (AppContext.jsx)
    # This is the SAME prompt given to BOTH agents
    STORY_COLLABORATOR_PROMPT = """You are an author taking part in a collaborative storytelling game activity with another author.
Together, you will create a story by taking turns adding to it. 
Your goal is to continue from where your partner has left off.
If there's no story, please begin the story. You have 10 interactions to write the story. Your input may get slightly truncated with a random character amount."""

    # The starting context that the first agent continues from
    STORY_PREFIX = "This is the story of "

    @staticmethod
    def get_system_prompt(turn_number: int = 1) -> str:
        """Get system prompt for any agent (both agents get identical prompts).
        
        Args:
            turn_number: Current turn number for pacing info
            
        Returns:
            System prompt with pacing metadata
        """
        base = SystemPrompts.STORY_COLLABORATOR_PROMPT
        # Add pacing info like the real experiment server does
        pacing = f"""

[Session Meta — do not reveal]
This is turn {turn_number} of 10. Use this only to pace and conclude appropriately. Do not mention turns, counts, chapters, headings, "#", or session meta in your reply. Write in plain prose that flows naturally from the previous text."""
        
        return f"{base}{pacing}".strip()


def simulate_single_story(
    provider_agent1: BaseProvider,
    provider_agent2: BaseProvider,
    n_turns: int,
    story_id: str = "story_1",
    model_id: str = "unknown",
    temperature: float = 1.0,
    max_tokens: int = 40,
    truncate_words_min: int = 2,
    truncate_words_max: int = 5,
) -> pd.DataFrame:
    """Simulate a single AI-AI story conversation with symmetric word-truncation.

    Mirrors the human-human experiment's symmetric structure: BOTH agents'
    outputs are word-truncated each turn (random 2..5 words dropped from
    the end), just as both humans' typed inputs were truncated by
    truncateUserInput in src/utils/submissionHelpers.js before being
    saved and sent to the partner.

    The truncated text is permanent — what's stored, what's added to each
    agent's history, and what's sent to the other agent as user input on
    the next turn. Both agents are computationally identical, so symmetric
    treatment avoids any within-condition role confound and matches HH.

    Per-agent context management:
    - Each agent maintains its own alternating user/assistant history.
    - OpenAI uses native conversation threading; Claude / HF receive the
      full rolling history + the current user input each call.
    - System prompt updated each turn with pacing metadata.

    Args:
        provider_agent1: Provider for first agent.
        provider_agent2: Provider for second agent.
        n_turns: Number of turns (each turn = agent1 + agent2 response).
        story_id: Unique story identifier.
        model_id: Model identifier for tracking.
        temperature: Sampling temperature (default 1.0 to match experiment).
        max_tokens: Max output tokens for both agents (default 40, roughly
            matches the experiment's USER_INPUT_LIMITS char range).
        truncate_words_min, truncate_words_max: Inclusive range for the
            random number of words trimmed from each agent's output each
            turn. Mirrors truncateUserInput; deployed JS uses
            Math.floor(Math.random()*5)+1 (1..5), set once per session.

    Returns:
        DataFrame with one row per turn (columns: turn, agent_1, agent_2,
        story_id, model_id, timestamp).
    """
    # Reset both providers for fresh conversation
    provider_agent1.reset_conversation()
    provider_agent2.reset_conversation()
    
    # Initialize conversation
    data = []
    
    # Each agent maintains its own conversation history as alternating
    # user/assistant messages, matching the experiment's frontend state.
    # From each agent's perspective:
    #   "user" = partner's text (what the other agent wrote)
    #   "assistant" = own response
    agent1_history: List[Dict[str, str]] = []
    agent2_history: List[Dict[str, str]] = []
    
    # Track full story text for data recording only
    story_context = SystemPrompts.STORY_PREFIX
    
    # Will hold agent 2's last response for next iteration
    agent2_last_text = ""
    
    for turn_idx in range(n_turns):
        turn_number = turn_idx + 1
        
        # BOTH agents get the SAME system prompt (matching actual experiment)
        system_prompt = SystemPrompts.get_system_prompt(turn_number)
        
        # === Agent 1's turn ===
        # Determine what agent 1 receives as "user input" from its partner
        if turn_idx == 0:
            # First turn: the baseline prefix is the starting context
            # (matches experiment: baselineText = "This is the story of")
            agent1_input = SystemPrompts.STORY_PREFIX
        else:
            # Subsequent turns: agent 2's last response is the partner input
            # (matches experiment: human types text -> becomes userInput for AI)
            agent1_input = agent2_last_text
        
        agent1_text = provider_agent1.generate(
            system_prompt=system_prompt,
            user_input=agent1_input,
            history=agent1_history if agent1_history else None,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Strip echo of partner input — matches experiment frontend
        # (src/services/aiService.jsx generateAIResponse), applied to all providers.
        agent1_text = strip_input_echo(agent1_text, agent1_input)

        # Symmetric word-truncation: drop 2..5 words from end (mirrors
        # truncateUserInput from src/utils/submissionHelpers.js, applied
        # to both agents per the symmetric HH design). Permanent — flows
        # into history, partner's next input, and the saved DataFrame.
        agent1_text = truncate_user_input(
            agent1_text,
            random.randint(truncate_words_min, truncate_words_max),
        )

        # Update agent 1's history (user = partner input, assistant = own response)
        agent1_history.append({"role": "user", "content": agent1_input})
        agent1_history.append({"role": "assistant", "content": agent1_text})
        
        # Update story context for data recording
        if turn_idx == 0:
            # First turn: agent 1 continued from "This is the story of"
            # Save with prefix for data (matching human-AI data structure)
            agent1_text_for_data = f"{SystemPrompts.STORY_PREFIX}\n{agent1_text}"
            story_context = agent1_text_for_data
        else:
            agent1_text_for_data = agent1_text
            story_context = f"{story_context} {agent1_text}"
        
        # === Agent 2's turn ===
        # Agent 2 receives agent 1's output as partner input
        # (matches experiment: AI response becomes context for next human turn,
        #  but here the "next human" is another AI)
        agent2_input = agent1_text
        
        agent2_text = provider_agent2.generate(
            system_prompt=system_prompt,
            user_input=agent2_input,
            history=agent2_history if agent2_history else None,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Strip echo of partner input — matches experiment frontend
        # (src/services/aiService.jsx generateAIResponse), applied to all providers.
        agent2_text = strip_input_echo(agent2_text, agent2_input)

        # Symmetric word-truncation: drop 2..5 words from end (mirrors
        # truncateUserInput from src/utils/submissionHelpers.js).
        agent2_text = truncate_user_input(
            agent2_text,
            random.randint(truncate_words_min, truncate_words_max),
        )

        # Update agent 2's history
        agent2_history.append({"role": "user", "content": agent2_input})
        agent2_history.append({"role": "assistant", "content": agent2_text})
        
        # Save agent 2's last text for next iteration
        agent2_last_text = agent2_text
        
        # Update story context for data recording
        story_context = f"{story_context} {agent2_text}"
        
        # Record turn data
        data.append({
            "turn": turn_number,
            "agent_1": agent1_text_for_data,
            "agent_2": agent2_text,
            "story_id": story_id,
            "model_id": model_id,
            "timestamp": pd.Timestamp.now()
        })
    
    return pd.DataFrame(data)


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 2.0):
    """Execute function with exponential backoff retry on rate limit errors.
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors (429) or similar
            is_rate_limit = (
                '429' in error_str or 
                'rate limit' in error_str or
                'rate_limit' in error_str or
                'too many requests' in error_str
            )
            
            if not is_rate_limit:
                raise  # Non-rate-limit error, don't retry
            
            last_exception = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            if attempt < max_retries - 1:
                print(f"  ⏳ Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  ❌ Rate limit persists after {max_retries} attempts")
    
    raise last_exception


def simulate_ai_ai_dataset(
    model_configs: List[Dict],
    n_stories_per_model: int = 10,
    n_turns_per_story: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 40,
    delay_between_stories: float = 2.0,
    delay_between_models: float = 5.0
) -> pd.DataFrame:
    """Generate full AI-AI simulation dataset with rate limit handling.

    Uses round-robin model rotation to distribute requests across providers,
    reducing rate limit issues. Includes retry logic with exponential backoff.

    Matches the symmetric structure of the PENPAL human-human condition:
    BOTH agents' outputs are word-truncated 2..5 words each turn before
    being saved AND sent to the other agent as its next user input.
    Mirrors truncateUserInput from src/utils/submissionHelpers.js, applied
    bilaterally because both AI-AI partners are computationally identical.

    - Temperature: 1.0
    - Max tokens: 40 (both agents)
    - 10 turns per story
    
    Args:
        model_configs: List of model configurations with keys:
            - id: Model identifier (e.g., 'gpt-4o')
            - provider: Provider type ('openai', 'anthropic', 'huggingface')
            - model_name: Full model name for API
            - env_key: Environment variable name for API key
            - base_url: (optional) Base URL for HuggingFace
        n_stories_per_model: Number of stories to generate per model
        n_turns_per_story: Number of turns per story
        temperature: Sampling temperature (default 1.0 to match experiment)
        max_tokens: Max output tokens (default 35 to match experiment)
        delay_between_stories: Seconds to wait between stories (default 2.0)
        delay_between_models: Seconds to wait when switching models (default 5.0)
        
    Returns:
        Combined DataFrame with all simulated stories
    """
    all_stories = []
    
    # Prepare model providers (skip those without API keys)
    active_models = []
    for model_config in model_configs:
        model_id = model_config['id']
        provider_type = model_config['provider']
        model_name = model_config['model_name']
        env_key = model_config['env_key']
        base_url = model_config.get('base_url')
        
        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"⚠️ Skipping {model_id}: {env_key} not set in environment")
            continue
        
        # Create providers for this model (both agents use same model)
        provider_agent1 = get_provider(provider_type, api_key, model_name, base_url)
        provider_agent2 = get_provider(provider_type, api_key, model_name, base_url)
        
        active_models.append({
            'config': model_config,
            'provider_agent1': provider_agent1,
            'provider_agent2': provider_agent2,
            'stories_generated': 0
        })
    
    if not active_models:
        print("\n❌ No models available. Check API keys.")
        return pd.DataFrame()
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Round-robin simulation across {len(active_models)} models")
    print(f"  Stories per model: {n_stories_per_model}")
    print(f"  Turns per story: {n_turns_per_story}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Delay between stories: {delay_between_stories}s")
    print(f"  Models: {', '.join(m['config']['id'] for m in active_models)}")
    print(f"{'='*60}")
    
    # Round-robin: generate one story per model at a time
    total_stories = n_stories_per_model * len(active_models)
    
    with tqdm(total=total_stories, desc="Total progress") as pbar:
        story_round = 0
        while any(m['stories_generated'] < n_stories_per_model for m in active_models):
            story_round += 1
            
            for model_idx, model_data in enumerate(active_models):
                if model_data['stories_generated'] >= n_stories_per_model:
                    continue  # This model is done
                
                model_config = model_data['config']
                model_id = model_config['id']
                story_num = model_data['stories_generated'] + 1
                story_id = f"{model_id}_story_{story_num}"
                
                # Reset providers for new story
                model_data['provider_agent1'].reset_conversation()
                model_data['provider_agent2'].reset_conversation()
                
                try:
                    # Use retry wrapper for rate limit handling
                    def generate_story():
                        return simulate_single_story(
                            provider_agent1=model_data['provider_agent1'],
                            provider_agent2=model_data['provider_agent2'],
                            n_turns=n_turns_per_story,
                            story_id=story_id,
                            model_id=model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    
                    df_story = retry_with_backoff(generate_story, max_retries=5, base_delay=2.0)
                    all_stories.append(df_story)
                    model_data['stories_generated'] += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"{model_id} {story_num}/{n_stories_per_model}")
                    
                except Exception as e:
                    print(f"\n  ❌ Error generating {story_id}: {e}")
                    model_data['stories_generated'] += 1  # Count as attempted
                    pbar.update(1)
                
                # Delay between stories (skip after last story)
                remaining = sum(n_stories_per_model - m['stories_generated'] for m in active_models)
                if remaining > 0:
                    time.sleep(delay_between_stories)
    
    if not all_stories:
        print("\n❌ No stories generated. Check API keys.")
        return pd.DataFrame()
    
    result = pd.concat(all_stories, ignore_index=True)
    print(f"\n✓ Generated {len(result)} turns across {len(all_stories)} stories")
    
    return result


# Legacy compatibility functions
def simulate_ai_ai_conversations(
    client_ids: List[str],
    workshop_ids: List[str],
    n_interactions: int,
    n_simulations: int,
    api_key: str
) -> pd.DataFrame:
    """Legacy function for backward compatibility.
    
    Deprecated: Use simulate_ai_ai_dataset() instead.
    """
    print("⚠️ Using legacy simulation function. Consider using simulate_ai_ai_dataset().")
    
    # Use single OpenAI model
    model_configs = [{
        'id': 'gpt-4o-mini',
        'provider': 'openai',
        'model_name': 'gpt-4o-mini',
        'env_key': 'OPENAI_API_KEY'
    }]
    
    # Temporarily set API key
    os.environ['OPENAI_API_KEY'] = api_key
    
    return simulate_ai_ai_dataset(
        model_configs=model_configs,
        n_stories_per_model=n_simulations,
        n_turns_per_story=n_interactions,
        temperature=1.0,
        max_tokens=35
    )
