"""Lightweight VADER-based sentiment scoring for news headlines.

Used at inference time so the Flask dashboard can accept a free-text headline
and blend it into the feature vector. The same scorer could be applied over a
dated news archive to replace the training-time proxy in features.build_features.
"""
from __future__ import annotations

from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """Returns VADER compound score in [-1, 1]. Empty text -> 0.0."""
    if not text or not text.strip():
        return 0.0
    return float(_analyzer().polarity_scores(text)["compound"])
