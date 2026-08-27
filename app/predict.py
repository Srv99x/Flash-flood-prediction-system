import pickle
from pathlib import Path
import pandas as pd

# RandomForest is the shipped model: it won on PR-AUC (0.8257 vs XGBoost's
# 0.7504) on the episode-grouped split. See docs/model_training_log.md for the
# split method, seed, and full metrics; rerun app/train_trigger_model.py to
# reproduce.
MODEL_PATH = Path("models/random_forest_trigger_model.pkl")
_model = None
_susceptibility_df = None

# We must ensure the features match exactly what XGBoost/RandomForest expects (reduced set)
EXPECTED_FEATURES = [
    "soil_moisture_0_7", "soil_moisture_7_28",
    "temp_c", "api_3d", "api_7d"
]

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

# ── Susceptibility class -> static risk multiplier ────────────────────
# TEAM-ASSIGNED weights. NOT derived from a cited study.
#
# The multiplier gates how much of the dynamic trigger probability reaches
# the final score, so it sets the risk scale on high-rainfall dates, where
# trigger probability saturates near 1.0 and final_risk ~= multiplier.
#
# Previous values were 0.1 / 0.3 / 0.7 / 1.0. They were raised because,
# combined with the old slope cutoffs, 73% of the district was multiplied
# by 0.1 and the map could not display a High or Severe cell on any date in
# the record: the single most extreme rainfall hour in eight years produced
# 770 Low / 133 Medium / 1 High / 0 Severe.
#
# Chosen so that on an extreme-rainfall date each susceptibility class lands
# in its own severity band (Low <0.25, Medium 0.25-0.50, High 0.50-0.75,
# Severe >=0.75), while a moderate monsoon date (trigger probability <= 0.09)
# still keeps every cell in Low.
SUSCEPTIBILITY_MULTIPLIERS = {
    "Low": 0.20,
    "Moderate": 0.45,
    "High": 0.70,
    "Very High": 0.90,
}


def get_susceptibility_multiplier(grid_ids: pd.Series) -> pd.Series:
    """
    Look up susceptibility class from pre-computed parquet and map it to a
    static risk multiplier.

    See SUSCEPTIBILITY_MULTIPLIERS above: the values are team-assigned and
    tunable, not empirical constants.  Cells with no DEM coverage get 0.0
    and are rendered as "No Data" (grey) by the Streamlit app rather than
    as genuine low risk.
    """
    global _susceptibility_df
    if _susceptibility_df is None:
        sus_path = Path("data/processed/susceptibility_features.parquet")
        if not sus_path.exists():
            raise FileNotFoundError(f"Susceptibility features not found at {sus_path}")
        _susceptibility_df = pd.read_parquet(sus_path)
    
    # Create lookup series indexed by grid_id
    lookup = (
        _susceptibility_df.set_index("grid_id")["gsi_susceptibility_class"]
        .map(SUSCEPTIBILITY_MULTIPLIERS)
    )
    
    # Map grid_ids to multipliers
    return grid_ids.map(lookup).fillna(0.0)

def predict_risk(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict landslide risk probabilities combining dynamic trigger and static susceptibility.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        DataFrame containing the required features for the model and 'grid_id'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - dynamic_trigger_prob: Raw probability from model (wetness)
        - susceptibility_mult: Static multiplier from terrain slope
        - final_risk_score: Combined gated probability
    """
    model = load_model()
    
    # Ensure all required features are present
    missing = [f for f in EXPECTED_FEATURES if f not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required features for prediction: {missing}")
    if "grid_id" not in feature_df.columns:
        feature_df = feature_df.reset_index()
        if "grid_id" not in feature_df.columns:
            raise ValueError("Missing 'grid_id' required for susceptibility layer.")

    # 1. Dynamic Trigger Layer
    X = feature_df[EXPECTED_FEATURES]
    dynamic_probs = model.predict_proba(X)[:, 1]
    
    # 2. Static Susceptibility Layer
    susceptibility_mult = get_susceptibility_multiplier(feature_df["grid_id"]).values
    
    # 3. Final Combined Score
    final_risk = dynamic_probs * susceptibility_mult
    
    result = pd.DataFrame({
        "grid_id": feature_df["grid_id"],
        "dynamic_trigger_prob": dynamic_probs,
        "susceptibility_mult": susceptibility_mult,
        "final_risk_score": final_risk
    })
    # Restore index if it was originally there
    if result.index.name != feature_df.index.name or not (result.index == feature_df.index).all():
        result.index = feature_df.index
        
    return result
