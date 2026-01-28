"""LLM Configuration - Groq with rate limit handling and retries.

Model: llama-3.3-70b-versatile (12K TPM - highest among Llama models)
Strategy: Add delays between calls to avoid rate limits
"""
import os
from crewai import LLM

# Best model for rate limits on free tier
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.3) -> LLM:
    """Get Groq LLM with rate limit optimization."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    return LLM(
        model=DEFAULT_MODEL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=1000,  # Increased for better responses
        timeout=120,
        max_retries=5,
    )
