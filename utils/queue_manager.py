import asyncio
import uuid
from typing import Callable, Any, Dict

tasks: Dict[str, Dict[str, Any]] = {}
queue: asyncio.Queue = asyncio.Queue()

async def add_task(func: Callable, *args, **kwargs) -> str:
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    await queue.put((task_id, func, args, kwargs))
    return task_id

async def worker():
    while True:
        task_id, func, args, kwargs = await queue.get()
        try:
            tasks[task_id]["status"] = "running"
            result = await asyncio.to_thread(func, *args, **kwargs)
            tasks[task_id] = {"status": "completed", "result": result}
        except Exception as e:
            tasks[task_id] = {"status": "error", "error": str(e)}
        finally:
            queue.task_done()

def get_task(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})
