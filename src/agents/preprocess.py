import re

from agents.base import BaseAgent
from core.state import PipelineState

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str, level: str = "full") -> str:
    if not isinstance(text, str):
        return ""
    text = _WS.sub(" ", text).strip()
    if level == "light":
        return text
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = re.sub(r"x{2,}", " ", text, flags=re.IGNORECASE)  # CFPB redactions
    return _WS.sub(" ", text).strip()


def is_gibberish(text: str) -> bool:
    """Returns True if the text is mostly symbols (like uuencoded binary files)."""
    if not text:
        return True
    # Count letters, numbers, and spaces
    alnum_spaces = sum(c.isalnum() or c.isspace() for c in text)
    # If less than 70% of the text is actual letters/numbers/spaces, it's gibberish
    return (alnum_spaces / len(text)) < 0.7


class PreprocessAgent(BaseAgent):
    name = "preprocess"

    def _run(self, state: PipelineState) -> PipelineState:
        df = state.df
        text_col = state.config["data"]["text_col"]
        label_col = state.config["data"]["label_col"]
        level = self.cfg.get("level", "full")
        min_len = self.cfg.get("min_chars", 50)

        df = df.dropna(subset=[text_col, label_col]).copy()
        df["clean_text"] = df[text_col].map(lambda t: clean_text(t, level))
        
        # Filter 1: Minimum length
        df = df[df["clean_text"].str.len() >= min_len]
        
        # Filter 2: Remove gibberish/binary files
        df = df[~df["clean_text"].apply(is_gibberish)]
        
        # Filter 3: Remove duplicates
        df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

        state.df = df
        state.texts = df["clean_text"].tolist()
        state.labels = df[label_col].astype(str).tolist()
        state.metrics["n_docs"] = len(df)
        return state