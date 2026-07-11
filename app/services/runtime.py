"""In-memory holder for the currently active dataset.

NOTE: This is process-global state carried over from the original single-user
design. Making it per-session/thread-safe is a follow-up task.
"""

import pandas as pd


class Runtime:
    """Holds the currently active dataset in memory."""

    def __init__(self):
        """Initialize with no dataset loaded."""
        # Latest query result; None until the first query runs.
        self.df: pd.DataFrame | None = None


runtime = Runtime()
