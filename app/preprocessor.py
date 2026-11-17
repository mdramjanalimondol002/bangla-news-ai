import re
import unicodedata

# Common Bengali stopwords that add noise to lexical classification
BANGLA_STOPWORDS = {
    "এবং", "ও", "আর", "কিন্তু", "তবে", "যদি", "তাহলে", "কারণ", "যে", "যা",
    "যিনি", "যারা", "তার", "তাদের", "তিনি", "সে", "তারা", "এই", "সেই", "এটি",
    "করে", "করা", "করেছেন", "করেন", "করতে", "হয়", "হয়", "হয়েছে", "হয়েছে",
    "হবে", "ছিল", "ছিলেন", "আছে", "আছেন", "নেই", "থেকে", "দিয়ে", "দিয়ে",
    "জন্য", "সঙ্গে", "সাথে", "পর্যন্ত", "মধ্যে", "উপর", "নিচে", "বলে", "বলেন",
    "জানান", "জানিয়েছেন", "জানিয়েছেন", "গত", "আজ", "কাল", "একটি", "দুটি"
}


def normalize_bangla_text(text: str) -> str:
    """Normalize Unicode characters and strip noise from Bengali news text."""
    if not text:
        return ""

    # Normalize Unicode (NFC ensures composed Bengali characters like য়, ড় are consistent)
    text = unicodedata.normalize("NFC", text)

    # Remove URLs, HTML tags, and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep only Bengali Unicode block (\u0980-\u09FF), basic Latin letters, and spaces
    text = re.sub(r"[^\u0980-\u09FFa-zA-Z\s]", " ", text)

    # Remove Bengali and English digits
    text = re.sub(r"[\u09E6-\u09EF0-9]+", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize_and_clean(text: str) -> str:
    """Normalize text and remove common Bengali stopwords."""
    cleaned = normalize_bangla_text(text)
    tokens = [tok for tok in cleaned.split() if tok not in BANGLA_STOPWORDS and len(tok) > 1]
    return " ".join(tokens)