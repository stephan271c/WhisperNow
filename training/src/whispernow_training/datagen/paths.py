"""Filesystem locations shared by every stage."""

from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = TRAINING_ROOT.parent
PROMPTS = TRAINING_ROOT / "prompts"
DATA = TRAINING_ROOT / "data"
CACHE = TRAINING_ROOT / ".cache"

STAGE1 = DATA / "stage1"
STAGE2 = DATA / "stage2"
STAGE3 = DATA / "stage3"
BUILD = DATA / "build"
# Personal dictations. Gitignored with the rest of data/, and never published or trained on.
REAL = DATA / "real"
REVIEW = DATA / "review"
