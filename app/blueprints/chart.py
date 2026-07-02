"""Chart data route (AJAX) — returns JSON for the frontend ECharts renderer."""

from flask import Blueprint, jsonify, request

from ..services.runtime import runtime
from ..services.visualization import build_chart_payload

chart_bp = Blueprint("chart", __name__)


@chart_bp.route("/generate_chart", methods=["POST"])
def generate_chart():
    chart_type = request.form.get("chart_type")

    if runtime.df is None or runtime.df.empty:
        return jsonify({"error": "No data available for visualization."})

    return jsonify(build_chart_payload(runtime.df, chart_type))
