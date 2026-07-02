"""In-memory holder for the currently active dataset and SQL agent.

NOTE: This is process-global state carried over from the original single-user
design. It is intentionally centralized here so the refactor preserves the
existing behavior; making it per-session/thread-safe is a follow-up task.
"""


class Runtime:
    def __init__(self):
        self.df = None          # latest query result as a pandas DataFrame
        self.sql_agent = None   # LangChain SQL agent bound to the active table


runtime = Runtime()
