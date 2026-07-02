"""In-memory holder for the currently active dataset.

NOTE: This is process-global state carried over from the original single-user
design. Making it per-session/thread-safe is a follow-up task.
"""


class Runtime:
    def __init__(self):
        self.df = None  # latest query result as a pandas DataFrame


runtime = Runtime()
