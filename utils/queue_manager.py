# utils/queue_manager.py
import asyncio
import uuid
from typing import Callable, Any
import traceback

tasks: dict = {}
queue: asyncio.Queue = asyncio.Queue()


async def add_task(func: Callable, *args, **kwargs) -> str:
    """
    Add a blocking function (or coroutine) to the queue.
    func should be a regular blocking function (not coroutine) OR an async function.
    Return task_id immediately.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    await queue.put((task_id, func, args, kwargs))
    return task_id


async def worker():
    """Continuously processes tasks from the queue. Uses asyncio.to_thread for blocking functions."""
    while True:
        task_id, func, args, kwargs = await queue.get()
        try:
            tasks[task_id]["status"] = "running"
            # If func is coroutinefunction, await it; else run in thread
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # run blocking function in threadpool
                result = await asyncio.to_thread(_call_safe, func, *args, **kwargs)

            tasks[task_id] = {"status": "done", "result": result}
        except Exception as e:
            tb = traceback.format_exc()
            tasks[task_id] = {"status": "error", "error": str(e), "traceback": tb}
        finally:
            queue.task_done()


def _call_safe(func, *args, **kwargs):
    """Wrapper to call synchronous functions and catch exceptions to propagate out."""
    return func(*args, **kwargs)


def get_task(task_id: str) -> dict:
    return tasks.get(task_id, {"status": "not_found"})
