"""
Shared figure styling for TWS downscaling outputs.

Conventions (per project style):
- All figures saved at 600 DPI.
- No underscores in any visible label, legend entry or title.
- R-squared rendered as a proper superscript (mathtext ``$R^2$`` in figures,
  the unicode ``R²`` in console / CSV text).
- Model and CV-scheme identifiers shown as clean, human-readable names.
"""

from __future__ import annotations

DPI = 600

# Matplotlib mathtext for axis labels / titles.
R2 = r"$R^2$"
# Plain-text (console, CSV, markdown) superscript.
R2_TXT = "R²"

# Canonical model keys (as used in prediction filenames) -> display names.
_MODEL_DISPLAY = {
    "RandomForest": "Random Forest",
    "randomforest": "Random Forest",
    "random_forest": "Random Forest",
    "XGBoost": "XGBoost",
    "xgboost": "XGBoost",
    "LightGBM": "LightGBM",
    "lightgbm": "LightGBM",
    "LSTM": "LSTM",
    "lstm": "LSTM",
    "BiLSTM": "BiLSTM",
    "bilstm": "BiLSTM",
    "BiLSTM+Attention": "BiLSTM + Attention",
    "bilstm_attention": "BiLSTM + Attention",
    "BiLSTM_Attention": "BiLSTM + Attention",
}

_SCHEME_DISPLAY = {
    "random_shuffled": "Random (shuffled)",
    "blocked": "Blocked",
    "purged_embargo": "Purged + embargo",
}


def pretty_model(name: str) -> str:
    """Human-readable model name with no underscores."""
    if name in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[name]
    return name.replace("_", " ").replace("+", " + ").strip()


def pretty_scheme(name: str) -> str:
    """Human-readable CV-scheme name with no underscores."""
    return _SCHEME_DISPLAY.get(name, name.replace("_", " "))


def clean_label(text: str) -> str:
    """Replace underscores and normalise R2 for any visible label."""
    return text.replace("_", " ").replace("R2", R2)


def clean_feature(name: str) -> str:
    """
    Display form for a lagged-feature name, e.g. 'SMS_lag1' -> 'SMS lag-1'.
    Keeps the underlying column name untouched; only affects what is drawn.
    """
    out = name.replace("_lag", " lag-").replace("_", " ")
    return out


def clean_features(names) -> list:
    """Vectorised `clean_feature` for a list/iterable of names."""
    return [clean_feature(n) for n in names]
