import os
from functools import lru_cache

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return Langfuse(tracing_enabled=enabled)
