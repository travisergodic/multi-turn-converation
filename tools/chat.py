import json
import os
import sys
import threading
import uuid
from pathlib import Path
from time import perf_counter

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 允许在任意目录执行 `python tools/chat.py` 时找到 `src` 与配置
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)
load_dotenv()

from src.graph import build_response_graph
from src.logging_utils import get_log_file, get_logger, setup_logging
from src.memory_store import FileMemoryStore
from src.nodes import memory_update_node, profile_update_node, summary_update_node

with open("configs/config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

COMMANDS = {
    "/new": "新建一个对话 thread",
    "/switch <id>": "切换到已有的 thread",
    "/threads": "列出本次会话中的所有 thread",
    "/quit": "退出",
}
THREADS_PATH = Path("data/threads.json")
setup_logging()
logger = get_logger(__name__)
SESSION_LOG_FILE = get_log_file()


def load_profile() -> dict:
    with open(_cfg["profile"]["path"], encoding="utf-8") as f:
        return json.load(f)


def load_known_threads() -> list[str]:
    if not THREADS_PATH.exists():
        return []
    try:
        with THREADS_PATH.open(encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Failed to decode threads file at %s", THREADS_PATH)
        return []
    return [thread_id for thread_id in payload if isinstance(thread_id, str)]


def save_known_threads(thread_ids: list[str]) -> None:
    THREADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = list(dict.fromkeys(thread_ids))
    with THREADS_PATH.open("w", encoding="utf-8") as f:
        json.dump(ordered_ids, f, ensure_ascii=False, indent=2)
    logger.info("Saved thread index path=%s count=%s", THREADS_PATH, len(ordered_ids))


def remember_thread(thread_ids: list[str], thread_id: str) -> list[str]:
    if thread_id not in thread_ids:
        thread_ids.append(thread_id)
        save_known_threads(thread_ids)
    return thread_ids


def print_help():
    print("\n可用命令：")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:22s} {desc}")
    print()


def run_background_updates(state: dict, store_path: str, thread_id: str) -> None:
    started_at = perf_counter()
    logger.info("Background updates started thread_id=%s", thread_id)
    store = FileMemoryStore(store_path)
    try:
        memory_update_node(state, store=store)
        updated_profile = profile_update_node(state, store=store)
        if updated_profile:
            state.update(updated_profile)
        updated_summary = summary_update_node(state, store=store)
        if updated_summary:
            state.update(updated_summary)
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info("Background updates finished thread_id=%s total_elapsed_ms=%.2f", thread_id, elapsed_ms)
    except Exception:
        logger.exception("Background updates failed thread_id=%s", thread_id)


def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    store_path = _cfg["memory"].get("store_path", "data/long_term_memory.json")
    graph = build_response_graph(store_path=store_path)
    user_id = "default_user"
    known_threads = load_known_threads()
    thread_id = known_threads[-1] if known_threads else str(uuid.uuid4())
    known_threads = remember_thread(known_threads, thread_id)
    logger.info(
        "CLI started user_id=%s current_thread=%s known_threads=%s session_log=%s",
        user_id,
        thread_id,
        len(known_threads),
        SESSION_LOG_FILE,
    )

    print(f"多轮对话系统已启动（当前 thread：{thread_id[:8]}...）")
    print("输入 /help 查看可用命令。\n")

    while True:
        try:
            user_input = input(f"[{thread_id[:8]}] 你：").strip()
        except (KeyboardInterrupt, EOFError):
            logger.info("CLI terminated by user input signal current_thread=%s", thread_id)
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            logger.info("CLI received quit command current_thread=%s", thread_id)
            print("再见！")
            break
        elif user_input == "/help":
            logger.info("CLI received help command current_thread=%s", thread_id)
            print_help()
            continue
        elif user_input == "/new":
            thread_id = str(uuid.uuid4())
            known_threads = remember_thread(known_threads, thread_id)
            logger.info("CLI created new thread thread_id=%s known_threads=%s", thread_id, len(known_threads))
            print(f"已新建 thread：{thread_id[:8]}...\n")
            continue
        elif user_input.startswith("/switch "):
            thread_id = user_input.split(maxsplit=1)[1].strip()
            known_threads = remember_thread(known_threads, thread_id)
            logger.info("CLI switched thread thread_id=%s known_threads=%s", thread_id, len(known_threads))
            print(f"已切换到 thread：{thread_id[:8]}...\n")
            continue
        elif user_input == "/threads":
            logger.info("CLI listed threads current_thread=%s known_threads=%s", thread_id, len(known_threads))
            for t in known_threads:
                marker = " <-- 当前" if t == thread_id else ""
                print(f"  {t[:8]}...{marker}")
            print()
            continue

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "retrieved_memories": [],
            "profile_snapshot": load_profile(),
        }

        logger.info("CLI invoking graph thread_id=%s user_input_chars=%s", thread_id, len(user_input))
        response_started_at = perf_counter()
        result = graph.invoke(state, config=config)
        response_elapsed_ms = (perf_counter() - response_started_at) * 1000
        ai_messages = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
        if ai_messages:
            logger.info(
                "CLI received assistant reply thread_id=%s reply_chars=%s response_elapsed_ms=%.2f",
                thread_id,
                len(ai_messages[-1].content or ""),
                response_elapsed_ms,
            )
            print(f"助手：{ai_messages[-1].content}\n")
            worker = threading.Thread(
                target=run_background_updates,
                args=(dict(result), store_path, thread_id),
                daemon=True,
                name=f"memory-maintenance-{thread_id[:8]}",
            )
            worker.start()
            logger.info("CLI scheduled background updates thread_id=%s", thread_id)


if __name__ == "__main__":
    main()
