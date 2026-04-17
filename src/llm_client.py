import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=_cfg["llm"]["base_url"],
)

DEFAULT_MODEL: str = _cfg["llm"]["model"]


def chat_completion(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
