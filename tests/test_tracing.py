import os


def test_get_langfuse_returns_instance_when_disabled():
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
    import src.eval.tracing as tracing
    lf = tracing.get_langfuse()
    assert lf is not None


def test_get_langfuse_singleton():
    import src.eval.tracing as tracing
    assert tracing.get_langfuse() is tracing.get_langfuse()
