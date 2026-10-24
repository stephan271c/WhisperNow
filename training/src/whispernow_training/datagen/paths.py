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
BUILD_V2 = DATA / "build-v2"
# build-v2 without the heard swaps, to measure what they add.
BUILD_V2_NO_HEARD = DATA / "build-v2-no-heard"
BUILD_V3 = DATA / "build-v3"
BUILD_V4 = DATA / "build-v4"
BUILD_V5 = DATA / "build-v5"
# Personal dictations. Gitignored with the rest of data/, and never published or trained on.
REAL = DATA / "real"
