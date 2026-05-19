"""
src/preprocessing/cleaner.py

Text cleaning and normalization pipeline for Hinglish social media text.

Handles:
  - Roman-script Hinglish  (e.g. "yaar ye movie bahut boring thi")
  - Mixed Roman + Devanagari (e.g. "yaar यह movie बहुत boring थी")
  - Social media noise: @mentions, URLs, hashtags, emoji, repeated chars
  - Transliteration normalization (optional)
"""

import re
import unicodedata
from typing import Optional

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False


# ─── Regex patterns (compiled once at module load) ────────────────────────────

_RE_URL       = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION   = re.compile(r"@\w+")
_RE_HASHTAG   = re.compile(r"#(\w+)")   # keep the word, drop the #
_RE_NUMBER    = re.compile(r"\b\d+\b")
_RE_REPEAT    = re.compile(r"(.)\1{2,}")  # "sooooo" → "soo" (keep 2)
_RE_SPACES    = re.compile(r"\s+")
_RE_PUNCT_RPT = re.compile(r"([!?.]){2,}")  # "!!!" → "!"

# Devanagari unicode range: \u0900–\u097F
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]+")

# Common Hinglish slang normalizations
_SLANG_MAP = {
    "yr":    "yaar",
    "yar":   "yaar",
    "bhai":  "bhai",
    "bro":   "bro",
    "nhi":   "nahi",
    "ni":    "nahi",
    "kyu":   "kyun",
    "kyun":  "kyun",
    "bcz":   "because",
    "bc":    "because",
    "tbh":   "to be honest",
    "imo":   "in my opinion",
    "lol":   "haha",
    "lmao":  "haha",
    "omg":   "oh my god",
    "wtf":   "what the",
    "idk":   "i don't know",
    "btw":   "by the way",
    "plz":   "please",
    "pls":   "please",
    "thx":   "thanks",
    "tysm":  "thank you so much",
    "ngl":   "not gonna lie",
    "fr":    "for real",
    "rn":    "right now",
    "irl":   "in real life",
    "smh":   "shaking my head",
    "fav":   "favourite",
    "pic":   "picture",
    "pics":  "pictures",
    "msg":   "message",
    "msgs":  "messages",
    "bakwas": "bakwas",
    "bekar":  "bekar",
    # ── Positive word variants (common misspellings) ──────────────────────
    "accha":  "accha",
    "acha":   "accha",
    "aachi":  "acchi",       # "aachi" → "acchi"
    "achi":   "acchi",
    "acchi":  "acchi",
    "achha":  "accha",
    "acha":   "accha",
    "bahut":  "bahut",
    "bhaut":  "bahut",       # "bhaut" → "bahut"  ← THIS was your bug
    "bahot":  "bahut",
    "bahat":  "bahut",
    "mast":   "mast",
    "zabardast": "zabardast",
    "zabardust": "zabardast",
    "kamaal": "kamaal",
    "kamal":  "kamaal",
    "shandar": "shandar",
    "shandaar": "shandar",
    # ── Common verb/connector variants ────────────────────────────────────
    "hai":    "hai",
    "he":     "hai",         # "he" → "hai"  ← THIS was also in your text
    "h":      "hai",
    "hain":   "hain",
    "tha":    "tha",
    "thi":    "thi",
    "nahi":   "nahi",
    "nahin":  "nahi",
    "mat":    "mat",
    "kya":    "kya",
    "kyaa":   "kya",
    "aur":    "aur",
    "or":     "aur",         # "or" as Hindi connector → "aur"
    # ── Common Hinglish intensifiers ──────────────────────────────────────
    "bilkul": "bilkul",
    "bilkull": "bilkul",
    "ekdum":  "ekdum",
    "ekdam":  "ekdum",
    "bohot":  "bahut",
    "boht":   "bahut",
    "bht":    "bahut",
}


class HinglishCleaner:
    """
    Pipeline for cleaning Hinglish social media text.

    Usage:
        cleaner = HinglishCleaner()
        clean_text = cleaner.clean("@RCB yaar ye bahut acchi innings thi!! 🔥🔥")
        # → "yaar ye bahut acchi innings thi !"
    """

    def __init__(
        self,
        remove_mentions: bool = True,
        expand_hashtags: bool = True,
        remove_urls: bool = True,
        handle_emoji: str = "remove",   # "remove" | "text" | "keep"
        normalize_slang: bool = True,
        normalize_repeats: bool = True,
        preserve_devanagari: bool = True,
        lowercase: bool = True,
        max_length: Optional[int] = None,
    ):
        self.remove_mentions    = remove_mentions
        self.expand_hashtags    = expand_hashtags
        self.remove_urls        = remove_urls
        self.handle_emoji       = handle_emoji
        self.normalize_slang    = normalize_slang
        self.normalize_repeats  = normalize_repeats
        self.preserve_devanagari = preserve_devanagari
        self.lowercase          = lowercase
        self.max_length         = max_length

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Unicode normalization (NFC: compose accented characters)
        text = unicodedata.normalize("NFC", text)

        # 2. Remove URLs
        if self.remove_urls:
            text = _RE_URL.sub(" ", text)

        # 3. Handle @mentions
        if self.remove_mentions:
            text = _RE_MENTION.sub(" ", text)

        # 4. Handle #hashtags — expand to word
        if self.expand_hashtags:
            text = _RE_HASHTAG.sub(r"\1", text)
        else:
            text = _RE_HASHTAG.sub(" ", text)

        # 5. Emoji handling
        if EMOJI_AVAILABLE:
            if self.handle_emoji == "text":
                # Convert 🔥 → "fire" etc.
                text = emoji.demojize(text, delimiters=(" ", " "))
                text = re.sub(r":[a-z_]+:", lambda m: m.group().replace("_", " ").strip(":"), text)
            elif self.handle_emoji == "remove":
                text = emoji.replace_emoji(text, replace=" ")
            # "keep" → do nothing
        else:
            # Fallback: remove emoji-range unicode chars
            text = re.sub(
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
                r"\U00002702-\U000027B0]+",
                " ", text
            )

        # 6. Lowercase (careful: preserve Devanagari — it's script-independent of case)
        if self.lowercase:
            # Lowercase only the ASCII/Latin portion, leave Devanagari intact
            if self.preserve_devanagari:
                parts = _RE_DEVANAGARI.split(text)
                devanagari_parts = _RE_DEVANAGARI.findall(text)
                lowercased = []
                for i, part in enumerate(parts):
                    lowercased.append(part.lower())
                    if i < len(devanagari_parts):
                        lowercased.append(devanagari_parts[i])
                text = "".join(lowercased)
            else:
                text = text.lower()

        # 7. Normalize slang
        if self.normalize_slang:
            words = text.split()
            words = [_SLANG_MAP.get(w, w) for w in words]
            text = " ".join(words)

        # 8. Normalize repeated characters: "soooo" → "soo"
        if self.normalize_repeats:
            text = _RE_REPEAT.sub(r"\1\1", text)
            text = _RE_PUNCT_RPT.sub(r"\1", text)

        # 9. Remove standalone numbers (often noise in tweets)
        text = _RE_NUMBER.sub(" ", text)

        # 10. Normalize whitespace
        text = _RE_SPACES.sub(" ", text).strip()

        # 11. Optional length truncation (character level, not token level)
        if self.max_length:
            text = text[: self.max_length]

        return text

    def clean_batch(self, texts: list) -> list:
        return [self.clean(t) for t in texts]


def detect_script_ratio(text: str) -> dict:
    """
    Return ratio of Devanagari vs Latin characters.
    Useful for understanding code-mixing level of a corpus.

    Returns:
        {"devanagari": 0.35, "latin": 0.60, "other": 0.05}
    """
    if not text:
        return {"devanagari": 0.0, "latin": 0.0, "other": 0.0}

    total = 0
    devanagari = 0
    latin = 0

    for ch in text:
        if ch.strip() == "":
            continue
        total += 1
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            devanagari += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            latin += 1

    if total == 0:
        return {"devanagari": 0.0, "latin": 0.0, "other": 0.0}

    return {
        "devanagari": round(devanagari / total, 3),
        "latin":      round(latin / total, 3),
        "other":      round((total - devanagari - latin) / total, 3),
    }


def is_hinglish(text: str, min_latin: float = 0.3, min_devanagari: float = 0.0) -> bool:
    """
    Heuristic check: does this text look like Hinglish?
    By default just checks that it has meaningful Latin content
    (most Hinglish is Roman-script).
    """
    ratios = detect_script_ratio(text)
    return ratios["latin"] >= min_latin


# ─── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleaner = HinglishCleaner(handle_emoji="text")

    samples = [
        "@RCB yaar ye match bahut bekar tha!! 😡😡😡",
        "accha movie thi bhai, must watch hai #Bollywood",
        "nhi yar, ye product bilkul bakwas hai. waste of money!!!",
        "यह बहुत accha tha, loved it so much 🔥",
        "https://t.co/xyz check this out!! lmao lol",
        "soooo boreddddd of this ngl fr fr",
    ]

    print("=" * 60)
    for s in samples:
        cleaned = cleaner.clean(s)
        ratio   = detect_script_ratio(cleaned)
        print(f"IN : {s}")
        print(f"OUT: {cleaned}")
        print(f"SCR: {ratio}")
        print("-" * 60)