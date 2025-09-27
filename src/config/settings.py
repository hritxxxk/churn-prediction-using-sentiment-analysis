"""
Configuration settings for the Netflix Churn Prediction System.
"""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

# Default file paths
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "small_sample.csv")
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.csv")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# NLP settings
MIN_WORD_LENGTH = 2
MAX_REVIEW_LENGTH = 512

# Clustering settings
PCA_VARIANCE_THRESHOLD = 0.95
N_CLUSTERS = 3

# Feature engineering thresholds
SATISFACTION_QUANTILE = 0.3
SENTIMENT_THRESHOLD = -0.3
SCORE_THRESHOLD = 2

# API Keys (should be set as environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")