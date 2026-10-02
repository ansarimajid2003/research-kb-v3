"""
Keyword-based strategy category classifier.
Returns (category, confidence) from paper title + abstract.
"""
from __future__ import annotations

# (keyword_fragments, category, weight)
_RULES: list[tuple[list[str], str, float]] = [
    # momentum
    (["momentum", "trend follow", "cta", "time-series momentum", "cross-sectional momentum",
      "relative strength", "trend strategy", "52-week high"], "momentum", 1.0),

    # mean-reversion
    (["mean reversion", "mean-reversion", "revert", "cointegrat", "pairs trading",
      "pair trading", "statistical arbitrage", "stat arb", "spread trading",
      "ornstein-uhlenbeck", "ou process"], "mean-reversion", 1.0),

    # carry
    (["carry trade", "carry strategy", "interest rate differential", "forward premium",
      "yield carry", "currency carry", "rolling yield"], "carry", 1.0),

    # volatility
    (["option pric", "volatility surface", "implied volatility", "vix", "volatility skew",
      "stochastic volatility", "delta hedg", "gamma", "vega", "option strategy",
      "variance swap", "volatility trading"], "volatility", 1.0),

    # arbitrage
    (["arbitrage", "market neutral", "long-short", "long short", "convergence trade",
      "merger arb", "convertible arb", "index arb", "etf arb"], "arbitrage", 1.0),

    # ml/alternative-data
    (["machine learning", "deep learning", "neural network", "lstm", "transformer",
      "reinforcement learning", "natural language processing", "nlp", "sentiment",
      "alternative data", "satellite", "credit card", "web scraping",
      "random forest", "gradient boost", "xgboost", "lightgbm"], "ml/alternative-data", 1.0),
]


def classify_paper(title: str, abstract: str) -> tuple[str, str]:
    """
    Returns (strategy_category, confidence).
    confidence: "high" (score >= 2), "medium" (score == 1), "low" (score < 1).
    Falls back to "" if nothing matches.
    """
    text = (title + " " + abstract).lower()
    scores: dict[str, float] = {}
    for keywords, category, weight in _RULES:
        for kw in keywords:
            if kw in text:
                scores[category] = scores.get(category, 0.0) + weight

    if not scores:
        return "", "low"

    best = max(scores, key=lambda k: scores[k])
    score = scores[best]
    confidence = "high" if score >= 2 else "medium" if score >= 1 else "low"
    return best, confidence
