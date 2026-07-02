"""Chart generation route (AJAX)."""

from flask import Blueprint, request

from ..services.runtime import runtime
from ..services.visualization import build_chart

chart_bp = Blueprint("chart", __name__)


@chart_bp.route("/generate_chart", methods=["POST"])
def generate_chart():
    chart_type = request.form.get("chart_type")

    if runtime.df is None or runtime.df.empty:
        return "<div class='alert alert-warning'>No data available for visualization.</div>"

    return build_chart(runtime.df, chart_type)
