"""Filesystem locations shared by every stage."""

from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = TRAINING_ROOT.parent
PROMPTS = TRAINING_ROOT / "prompts"
DATA = TRAINING_ROOT / "data"
CACHE = TRAINING_ROOT / ".cache"

AAWAAZ = DATA / "aawaaz"
PUBLIC = DATA / "public"
GENERATED = DATA / "generated"
BENCH = DATA / "bench"
# Each round's dataset; the last round writes `BUILD`.
ROUNDS = DATA / "rounds"
ROUND1 = ROUNDS / "1-base"
ROUND2 = ROUNDS / "2-targeted"
ROUND3 = ROUNDS / "3-weak-spots"
BUILD = DATA / "build"
# Personal dictations. Gitignored with the rest of data/, and never published or trained on.
REAL = DATA / "real"
