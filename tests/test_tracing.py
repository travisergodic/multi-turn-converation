import os
import pytest


@pytest.fixture(autouse=True)
def clear_langfuse_cache():
    import src.eval.tracing as tracing
    tracing.get_langfuse.cache_clear()
    yield
    tracing.get_langfuse.cache_clear()


def test_get_langfuse_returns_instance_when_disabled():
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
    import src.eval.tracing as tracing
    lf = tracing.get_langfuse()
    assert lf is not None
    assert lf._tracing_enabled is False


def test_get_langfuse_singleton():
    import src.eval.tracing as tracing
    assert tracing.get_langfuse() is tracing.get_langfuse()
