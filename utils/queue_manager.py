# utils/queue_manager.py
import asyncio
import uuid
import traceback

tasks = {}
queue = asyncio.Queue()


async def add_task(func, *args, **kwargs) -> str:
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    await queue.put((task_id, func, args, kwargs))
    return task_id


async def worker():
    while True:
        task_id, func, args, kwargs = await queue.get()
        try:
            tasks[task_id]["status"] = "running"

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            tasks[task_id] = {"status": "done", "result": result}

        except Exception as e:
            tb = traceback.format_exc()
            tasks[task_id] = {
                "status": "error",
                "error": str(e),
                "traceback": tb,
            }
        finally:
            queue.task_done()


def get_task(task_id: str) -> dict:
    return tasks.get(task_id, {"status": "not_found"})
