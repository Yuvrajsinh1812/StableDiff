# utils/queue_manager.py
import asyncio
import uuid
import time
from typing import Any

_tasks = {}
_queue = asyncio.Queue()
_worker_task = None
_worker_lock = asyncio.Lock()

async def add_task(func, *args, **kwargs) -> str:
    """
    Enqueue a callable (sync function). Returns a task_id.
    The callable is executed in the event loop (synchronously). If your function blocks
    for long (like model inference), it's still run in the same loop. This design is
    simple; if you need heavy concurrency consider running in a ThreadPoolExecutor.
    """
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "queued", "created_at": time.time()}
    await _queue.put((task_id, func, args, kwargs))
    return task_id

async def worker():
    """
    Consume queue forever, running work serially.
    """
    global _worker_task
    log_prefix = "[queue_worker]"
    while True:
        task_id, func, args, kwargs = await _queue.get()
        try:
            _tasks[task_id]["status"] = "running"
            # Run the function; if it raises, capture exception
            result = func(*args, **kwargs)
            _tasks[task_id] = {"status": "done", "result": result}
        except Exception as e:
            _tasks[task_id] = {"status": "error", "error": str(e)}
        finally:
            _queue.task_done()

def get_task(task_id: str) -> dict:
    return _tasks.get(task_id, {"status": "not_found"})

def start_worker_if_not_running(loop=None):
    """
    If worker isn't running, create an asyncio task for it.
    Call this from app startup to ensure only one worker exists.
    """
    global _worker_task
    if _worker_task is None or _worker_task.done():
        loop = loop or asyncio.get_event_loop()
        _worker_task = loop.create_task(worker())
