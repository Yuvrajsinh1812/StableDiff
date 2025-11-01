import asyncio
import uuid

tasks = {}
queue = asyncio.Queue()

async def add_task(func, *args, **kwargs):
    """Adds a new task to the queue."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    await queue.put((task_id, func, args, kwargs))
    return task_id

async def worker():
    """Continuously processes tasks from the queue."""
    while True:
        task_id, func, args, kwargs = await queue.get()
        try:
            tasks[task_id]["status"] = "running"

            # ✅ Await coroutine if func is async
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run blocking function in thread pool to prevent blocking event loop
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)

            tasks[task_id] = {"status": "done", "result": result}

        except Exception as e:
            tasks[task_id] = {"status": "error", "error": str(e)}

        finally:
            queue.task_done()

def get_task(task_id):
    """Returns the status or result of a task."""
    return tasks.get(task_id, {"status": "not_found"})
