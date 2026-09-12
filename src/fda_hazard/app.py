"""Server-rendered dashboard: FastAPI + HTMX + Plotly. No React, no build step.

Every figure and every number on these pages comes from queries.py, which reads
rows the evaluation wrote. Nothing is recomputed at render time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import queries as q
from .db import LABEL_CLASSES, connect
from .metrics import COST_MATRIX
from .splits import SPLIT_CUTOFF, Example

TEMPLATES = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES))

app = FastAPI(title="FDA recall hazard classification")

# Built lazily: the precedent index takes a few seconds to construct and only
# the single-case page needs it.
_index_cache: dict[str, Any] = {}


def db() -> sqlite3.Connection:
    return connect()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


templates.env.filters["fmt"] = _fmt
templates.env.filters["pct"] = lambda v: "-" if v is None else f"{v * 100:.1f}%"
templates.env.globals["LABEL_CLASSES"] = LABEL_CLASSES
templates.env.globals["COST_MATRIX"] = COST_MATRIX


def render(request: Request, template: str, **kw: Any) -> HTMLResponse:
    """Single render path. Current Starlette takes the request first; keeping
    that in one place stops the call sites from drifting apart."""
    return templates.TemplateResponse(
        request, template, {"cutoff": SPLIT_CUTOFF, **kw}
    )


# ---------------------------------------------------------------- overview


@app.get("/", response_class=HTMLResponse)
def overview(request: Request):
    conn = db()
    try:
        board = q.scoreboard(conn)
        agent = next((r for r in board if r["is_agent"]), None)
        headline = agent or (board[0] if board else None)

        confusion = None
        by_year: list[dict] = []
        if headline:
            row = q.run_by_uid(conn, headline["run_uid"])
            confusion = q.run_metrics(row)["confusion"]
            by_year = q.accuracy_by_year(conn, headline["run_uid"])

        dist = q.corpus_class_distribution(conn)
        return render(
            request,
            "overview.html",
            board=board,
            headline=headline,
            confusion=confusion,
            confusion_json=json.dumps(confusion) if confusion else "null",
            by_year=by_year,
            by_year_json=json.dumps(by_year),
            dist=dist,
            dist_json=json.dumps(dist),
            split=q.split_distribution(conn, SPLIT_CUTOFF),
            has_agent=agent is not None,
        )
    finally:
        conn.close()


# -------------------------------------------------------------------- runs


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    conn = db()
    try:
        rows = q.all_runs(conn)
        enriched = [{"row": r, "diff": q.run_diff(conn, r)} for r in rows]
        return render(request, "runs.html", runs=enriched)
    finally:
        conn.close()


# ------------------------------------------------------------------ errors


@app.get("/errors", response_class=HTMLResponse)
def errors_page(
    request: Request,
    run: str | None = None,
    sort: str = "cost",
    truth: str | None = None,
    predicted: str | None = None,
):
    conn = db()
    try:
        run_row = q.run_by_uid(conn, run) if run else q.latest(conn, "agent")
        if run_row is None:
            # No agent run yet: fall back to the first baseline that exists, so
            # the page is useful before the agent has ever been run.
            for system in q.BASELINE_SYSTEMS:
                run_row = q.latest(conn, system)
                if run_row is not None:
                    break
        if run_row is None:
            return render(
                request, "errors.html", run=None, rows=[], summary=None,
                runs=[], sort=sort, truth=truth, predicted=predicted,
            )

        rows = q.errors(
            conn, run_row["run_uid"], sort=sort, truth=truth, predicted=predicted
        )
        return render(
            request,
            "errors.html",
            run=run_row,
            rows=rows,
            summary=q.error_summary(conn, run_row["run_uid"]),
            runs=q.all_runs(conn, 50),
            sort=sort,
            truth=truth,
            predicted=predicted,
        )
    finally:
        conn.close()


@app.get("/errors/rows", response_class=HTMLResponse)
def errors_rows(
    request: Request,
    run: str,
    sort: str = "cost",
    truth: str | None = None,
    predicted: str | None = None,
):
    """HTMX partial: re-render just the error list on a sort or filter change."""
    conn = db()
    try:
        rows = q.errors(
            conn, run, sort=sort, truth=truth or None, predicted=predicted or None
        )
        return render(request, "_error_rows.html", rows=rows, run_uid=run)
    finally:
        conn.close()


# ------------------------------------------------------------- single case


@app.get("/case", response_class=HTMLResponse)
def case_form(request: Request):
    from .llm import available_providers

    return render(
        request, "case.html", result=None, providers=available_providers()
    )


@app.post("/case/run", response_class=HTMLResponse)
def case_run(
    request: Request,
    product_description: str = Form(""),
    reason_for_recall: str = Form(""),
    k: str = Form("5"),
):
    """Run the live agent on a pasted recall."""
    from .agent import HazardAgent
    from .llm import LLMClient, NoProviderConfigured, resolve_provider
    from .retrieval import build_index

    conn = db()
    try:
        if not reason_for_recall.strip() and not product_description.strip():
            return render(
                request,
                "_case_result.html",
                error="Enter a product description or a reason for recall.",
            )

        try:
            k_int = max(1, min(int(k), 10))
        except (TypeError, ValueError):
            k_int = 5

        example = Example(
            record_key="__live__",
            event_id=None,
            recalling_firm=None,
            product_description=product_description.strip() or "(not provided)",
            reason_for_recall=reason_for_recall.strip() or "(not provided)",
            classification="Class II",  # placeholder; a live case is never scored
            recall_initiation_date=None,
            distribution_pattern=None,
            voluntary_mandated=None,
            status=None,
        )

        try:
            provider = resolve_provider()
        except NoProviderConfigured as exc:
            return render(request, "_case_result.html", error=str(exc))

        if "index" not in _index_cache:
            _index_cache["index"] = build_index(conn, SPLIT_CUTOFF)

        client = LLMClient(conn, provider=provider)
        agent = HazardAgent(client, _index_cache["index"], conn, default_k=k_int)
        try:
            result = agent.classify(example)
        except Exception as exc:  # noqa: BLE001 - surfaced in the page, not swallowed
            return render(
                request,
                "_case_result.html",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            agent.close()
            client.close()

        cited = set(result.precedents_cited)
        precedents = [
            {**p, "cited": p.get("record_key") in cited}
            for p in result.precedents_seen
        ]
        return render(
            request,
            "_case_result.html",
            error=None,
            result=result,
            precedents=precedents,
            provider=provider.name,
            model=client.model,
        )
    finally:
        conn.close()
