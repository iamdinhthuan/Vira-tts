import re
import gc
import torch

# Lazy load normalizer to avoid import error if not installed
_normalizer = None

def get_normalizer():
    """Lazy load SoeNormalizer."""
    global _normalizer
    if _normalizer is None:
        try:
            from soe_vinorm import SoeNormalizer
            _normalizer = SoeNormalizer()
        except ImportError:
            print("Warning: soe_vinorm not installed. Text normalization disabled.")
            _normalizer = False
    return _normalizer


def fix_punctuation_spacing(text: str) -> str:
    """
    Fix spacing around punctuation marks.
    Remove space before punctuation, ensure space after.
    """
    # Remove space before punctuation marks
    text = re.sub(r'\s+([.,!?;:"])', r'\1', text)

    # Ensure space after punctuation (except at end of string)
    text = re.sub(r'([.,!?;:])(?=[^\s\d])', r'\1 ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese text using soe_vinorm.
    Converts numbers, abbreviations, etc. to spoken form.
    """
    normalizer = get_normalizer()

    if normalizer and normalizer is not False:
        text = normalizer.normalize(text)
        # Fix spacing issues from normalizer
        text = fix_punctuation_spacing(text)

    return text


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "Bạn cần nhập văn bản để tôi đọc."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def split_text(text: str, normalize: bool = True) -> list[str]:
    """
    Normalize punctuation and split text into sentences by . ! ?

    Args:
        text: Input text
        normalize: Whether to apply Vietnamese normalization (numbers, etc.)
    """
    # Apply Vietnamese normalization first (numbers -> words, etc.)
    if normalize:
        text = normalize_vietnamese(text)

    # Normalize punctuation
    text = punc_norm(text)

    # Split by sentence enders (. ! ?) while keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up each sentence
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
