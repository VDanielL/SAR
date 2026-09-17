"""
Manual (user-set) parameters for Sustained Anomaly Recognition (SAR).
"""

# Column names expected in the input CSV.
TIME_COLUMN = "timestamp"
VALUE_COLUMN = "value"

# T: periodicity of the input series, in seconds. The period in timesteps
# (P) is derived automatically from T and the inferred sampling interval.
PERIOD_SECONDS = 24 * 60 * 60  # 1 day

# W: width (in timesteps) of the moving-average window used to smooth the
# raw distance-to-pattern into the deviation score.
SMOOTHING_WINDOW = 12

# LAMBDA: sensitivity multiplier of the dynamic threshold
# Theta = mean(s) + LAMBDA * std(s).
LAMBDA = 3.0

# G_MAX: maximum gap (in timesteps) between two runs for them to be merged.
G_MAX = 3

# D_MIN: minimum duration (in timesteps) for a merged run to be kept.
D_MIN = 5
