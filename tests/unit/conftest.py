"""Unit-test defaults: never run the per-device kernel autotuning implicitly (tests/unit/test_autotune.py covers it)."""
import os

os.environ.setdefault("SUC_AUTOTUNE", "0")
