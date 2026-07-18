"""Chart data route (AJAX) — returns JSON for the frontend ECharts renderer."""

import logging

import pandas as pd
from flask import Blueprint, jsonify, request

from ..extensions import cache
from ..services.runtime import runtime
from ..services.visualization import apply_chart_spec, generate_chart_spec

logger = logging.getLogger("t2v.chart")

chart_bp = Blueprint("chart", __name__)


@chart_bp.route("/generate_chart", methods=["POST"])
def generate_chart():
    """Build a chart from a natural-language description of the active result.

    Accepts `description` (what to plot) and `scope` ("result" to
    subset/aggregate the result, "whole" to plot it as-is).
    """
    description = (request.form.get("description") or "").strip()
    scope = request.form.get("scope", "result")

    df = runtime.df
    if df is None or df.empty:
        return jsonify({"error": "No data available for visualization."})
    if not description:
        return jsonify({"error": "Describe the chart you'd like to see."})

    # Cache by result contents (hash ignores column labels) plus names, so
    # look-alike results don't collide.
    data_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    fingerprint = f"{list(df.columns)}:{data_hash}:{scope}:{description.lower()}"
    payload = cache.get(fingerprint)
    if payload is not None:
        return jsonify(payload)

    try:
        spec = generate_chart_spec(df, description, scope)
        payload = apply_chart_spec(df, spec)
    except Exception:
        logger.exception("Chart build failed | scope=%s | request=%r", scope, description)
        return jsonify({"error": "Couldn't build that chart. Try rephrasing your request."})

    if "error" not in payload:
        cache.set(fingerprint, payload)
    return jsonify(payload)