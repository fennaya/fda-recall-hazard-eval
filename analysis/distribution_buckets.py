"""Step 4: bucket recalls by distribution breadth, for error-slice analysis only.

`distribution_pattern` is free text from firms, not a controlled vocabulary
(seen: "US Nationwide.", "Nationwide within the U.S", "DE and NC", "MA", ...).
This module documents one fixed, unweighted rule for turning that text into
{nationwide, multi_state, single_state, other} and is used ONLY to slice
already-stored predictions after the fact. It is not, and must never become,
a new model input -- note that distribution_pattern was already part of the
agent's prompt in the run being analysed (see agent.py's USER_TEMPLATE), so
this script adds no new information the agent didn't already have; it only
groups existing, already-scored predictions for reporting.

The rule is fixed before looking at any error rates and is not adjusted to
change the result -- it is a categorisation scheme, not a tuned parameter.
"""

from __future__ import annotations

import re

# USPS two-letter codes, including DC and the territories that appear in this
# corpus (PR, VI, GU).
US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU",
}

_TOKEN_RE = re.compile(r"\b[A-Za-z]{2}\b")


def bucket_distribution(text: str | None) -> str:
    """Rule, applied in this fixed order:

    1. None/empty -> 'other' (nothing to categorise).
    2. Contains "nationwide" (any case) -> 'nationwide'. Covers the dozens of
       punctuation/wording variants seen in this corpus.
    3. Otherwise, extract standalone 2-letter tokens and count how many are
       valid US state/territory codes (case-insensitive).
       0 matches -> 'other' (e.g. a country name, "international", garbled text)
       1 match   -> 'single_state'
       2+ matches -> 'multi_state'
    """
    if not text:
        return "other"
    if "nationwide" in text.lower():
        return "nationwide"
    tokens = {t.upper() for t in _TOKEN_RE.findall(text)}
    hits = tokens & US_STATE_CODES
    if len(hits) == 0:
        return "other"
    if len(hits) == 1:
        return "single_state"
    return "multi_state"


if __name__ == "__main__":
    # Self-check against the earlier observed top distribution_pattern values,
    # printed for review, not asserted against any metric.
    samples = [
        "Nationwide in the USA", "US Nationwide.", "U.S. Nationwide",
        "Nationwide within the United States", "DE and NC", "MA",
        "CA, CO, FL, PR, WA", "US Nationwide , Alaska, and Puerto Rico.",
        "Nationwide in the USA and Antigua", None, "",
    ]
    for s in samples:
        print(f"{s!r:55} -> {bucket_distribution(s)}")
