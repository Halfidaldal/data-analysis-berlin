from typing import List, Optional
from pathlib import Path
import openai
import pandas as pd
import time
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

# Function to correct spelling
NaS = 0

def correct_spelling(text, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    global NaS  # Ensure global variable access
    if pd.isna(text) or not isinstance(text, str):
        NaS += 1
        return text  # Return original if NaN or not a string
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that ONLY corrects spelling mistakes while keeping the original meaning and structure intact. Output only the corrected text without any explanations."},
            {"role": "user", "content": text}
        ],
        temperature=0.0  # Ensures minimal alteration$
    )
    return response.choices[0].message.content


def compute_edit_distance_values(original_text, corrected_text):
    """Compute Levenshtein distance directly from two text values."""
    if pd.isna(original_text) or pd.isna(corrected_text):
        return None
    return levenshtein_distance(str(original_text), str(corrected_text))
