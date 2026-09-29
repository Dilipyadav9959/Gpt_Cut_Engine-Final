"""Unicode comparison keys preserving the vowel marks of Indic scripts."""

import unicodedata


def normalize_token(s: str) -> str:
    text = unicodedata.normalize("NFC", s).strip().casefold()
    # Python's \w excludes combining marks, including Telugu/Hindi vowel signs.
    # Preserve letters, marks and numbers. Separate at punctuation so "can-not"
    # cannot silently become the different token "cannot".
    characters = (char if unicodedata.category(char)[0] in "LMN" else " " for char in text)
    return " ".join("".join(characters).split())


def token_key(s: str) -> str:
    return normalize_token(s)
