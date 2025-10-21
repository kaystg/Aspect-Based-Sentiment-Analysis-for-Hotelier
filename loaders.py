from __future__ import annotations
import os, io, pathlib, typing as t
import streamlit as st
import pandas as pd
import joblib, requests, gdown

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = pathlib.Path("models"); MODEL_DIR.mkdir(exist_ok=True)

def _drive_download(file_id: str, out_path: pathlib.Path) -> pathlib.Path:
    """Download a public Google Drive file by ID (cached to disk)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out_path), quiet=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to download: {file_id}")
    return out_path

def _get_file_id(key: str) -> str:
    ids = st.secrets.get("drive_ids", {})
    if key not in ids:
        raise KeyError(f"Missing drive_ids['{key}'] in secrets.toml")
    return ids[key]

@st.cache_data(show_spinner=False)
def load_csv_from_drive(key: str, filename: str) -> pd.DataFrame:
    """Load CSV by drive_ids[key] → cache as DataFrame."""
    file_id = _get_file_id(key)
    local = DATA_DIR / filename
    _drive_download(file_id, local)
    return pd.read_csv(local)

@st.cache_resource(show_spinner=False)
def load_joblib_from_drive(key: str, filename: str):
    """Load joblib model by drive_ids[key] → cache as resource."""
    file_id = _get_file_id(key)
    local = MODEL_DIR / filename
    _drive_download(file_id, local)
    return joblib.load(local)

def ensure_nltk_packages(pkgs: t.Iterable[str] = ("punkt", "stopwords", "wordnet")) -> None:
    """Only download NLTK data when missing (keeps Streamlit logs clean)."""
    import nltk
    for p in pkgs:
        try:
            nltk.data.find(f"tokenizers/{p}" if p == "punkt" else f"corpora/{p}")
        except LookupError:
            nltk.download(p, quiet=True)
