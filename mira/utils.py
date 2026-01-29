import re
import gc
import torch


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

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


def split_text(text: str) -> list[str]:
    """
    Normalize punctuation and split text into sentences by . ! ?
    """
    # Normalize punctuation first
    text = punc_norm(text)

    # Split by sentence enders (. ! ?) while keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up each sentence
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
