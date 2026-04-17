import os
from time import perf_counter

import yaml
from openai import OpenAI
from dotenv import load_dotenv

from src.logging_utils import get_logger

load_dotenv()
logger = get_logger(__name__)

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=_cfg["llm"]["base_url"],
)

DEFAULT_MODEL: str = _cfg["llm"]["model"]


def chat_completion(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    started_at = perf_counter()
    logger.info("Calling chat completion with model=%s message_count=%s", model, len(messages))
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "Chat completion succeeded model=%s response_chars=%s elapsed_ms=%.2f",
        model,
        len(content or ""),
        elapsed_ms,
    )
    return content
