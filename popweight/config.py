"""Global configuration for the PopWeight pipeline."""

DATA_PATH = "data/social_media_engagement_data.xlsx"
SHEET_NAME = "Working File"
SQLITE_PATH = "outputs/results.sqlite"

RANDOM_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_RATIO = 0.8
TREND_PERCENTILE = 0.9

# Outlier controls
REMOVE_TOP_REACH_PERCENTILE = 0.995
MIN_REACH = 1

# Feature transform controls
USE_DOUBLE_LOG = True

# Segment definition
SEGMENT_KEYS = ["Platform", "Post Type"]
MIN_SEGMENT_SAMPLES = 20
