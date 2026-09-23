"""Hard token ceiling for a run, enforced in the runner itself.

Not a provider dashboard setting: this counts tokens the harness has actually
sent and received, in-process, and stops the run the moment the ceiling is
crossed -- so a paid key can never be charged more than a number decided
before the run started, independent of anything a provider's own billing
console does or does not enforce.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(BaseException):
    """Raised the instant cumulative spend reaches the ceiling.

    Deliberately subclasses BaseException, not Exception. Both
    HazardAgent.classify() and agent_single.classify_single() catch
    `except Exception` broadly, on purpose, so that one bad case (a
    malformed response, a transient provider error) never kills a run
    spanning hours. A budget cap is a different kind of signal: it must
    terminate the whole run, not be swallowed as a per-case fallback and
    let the run keep spending past the ceiling. Subclassing BaseException
    makes that true without editing either agent's exception handling.
    """

    def __init__(self, spent: int, ceiling: int):
        self.spent = spent
        self.ceiling = ceiling
        super().__init__(
            f"token budget exceeded: spent {spent:,} >= ceiling {ceiling:,}"
        )


@dataclass
class TokenBudget:
    """Tracks cumulative real (non-cache-hit) token spend across a run.

    `ceiling=None` means no cap -- used for runs where a cap doesn't apply
    (e.g. free-tier verification with nothing at financial stake). Thread-safe
    so the same instance can be shared across per-worker LLMClients.
    """

    ceiling: int | None
    log_path: Path | str | None = None
    spent: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        """Record a real call's tokens. Raises BudgetExceeded if this call
        pushes cumulative spend to or past the ceiling."""
        tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        with self._lock:
            self.spent += tokens
            self.calls += 1
            spent, ceiling = self.spent, self.ceiling
        if ceiling is not None and spent >= ceiling:
            raise BudgetExceeded(spent, ceiling)

    def log_case(self, cases_done: int, cases_total: int) -> None:
        """Append the running total to the progress log. Called once per
        completed case, not once per model call.

        A run that has already paid for real, rate-limited model calls must
        not lose its results to something as avoidable as a missing log
        directory: the parent directory is created if absent, same as
        db.connect() already does for the database file.
        """
        with self._lock:
            spent, calls, ceiling = self.spent, self.calls, self.ceiling
        if self.log_path is None:
            return
        pct = f"  ({100 * spent / ceiling:.1f}% of ceiling)" if ceiling else ""
        line = (
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
            f"case {cases_done}/{cases_total}  calls={calls}  "
            f"tokens_spent={spent:,}{pct}\n"
        )
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
