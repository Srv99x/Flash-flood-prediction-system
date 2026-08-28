"""
Shared configuration constants for the Flash Flood Risk system.

Single source of truth for values that are used by more than one module
(prediction, explanation, and the offline cache builder).  Keeping them here
prevents the copies from silently drifting out of sync.
"""

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
