"""Convenience re-exports for local maxima extraction (see utils/chm.py)."""

from utils.chm import extract_local_maxima, generate_chm, apply_closing_filter

__all__ = ["extract_local_maxima", "generate_chm", "apply_closing_filter"]
