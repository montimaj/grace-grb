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

# Matplotlib mathtext for axis labels / titles. \mathrm keeps the R UPRIGHT
# (a plain $R^2$ renders the R in maths italic, which is not wanted here).
R2 = r"$\mathrm{R}^{2}$"
# Bold-weight variant for use inside BOLD titles: matplotlib mathtext does NOT
# inherit a title's fontweight, so \mathrm renders normal-weight next to bold
# words. \mathbf keeps R and the exponent upright AND bold, matching the title.
R2_BOLD = r"$\mathbf{R^{2}}$"
# Plain-text (console, CSV, markdown) superscript.
R2_TXT = "R²"

# Scientific, colour-blind-safe categorical palette (publication quality).
# Blue / orange / aqua lead; this ordering is validated all-pairs for CVD.
SCI_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SCI_BLUE, SCI_ORANGE, SCI_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SCI_INK = "#0b0b0b"       # primary text
SCI_MUTED = "#898781"     # axes / muted labels
SCI_GRID = "#e1e0d9"      # hairline gridlines

# Seaborn "deep" default palette leads (blue, orange, green). Used as a single,
# consistent scheme for the error-metric bar plots: a single series is blue,
# a two-category contrast (train/test, default/tuned) is blue vs orange, and a
# three-category contrast (holdout, CV scheme) is blue/orange/green.
SB_BLUE, SB_ORANGE, SB_GREEN = "#4C72B0", "#DD8452", "#55A868"
BAR = SB_BLUE                                    # single series
BAR_2 = [SB_BLUE, SB_ORANGE]                     # two ordered categories
BAR_3 = [SB_BLUE, SB_ORANGE, SB_GREEN]           # three ordered categories

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
