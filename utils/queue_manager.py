import asyncio
import uuid

tasks = {}
queue = asyncio.Queue()

async def add_task(func, *args, **kwargs):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}
    await queue.put((task_id, func, args, kwargs))
    return task_id

async def worker():
    while True:
        task_id, func, args, kwargs = await queue.get()
        try:
            tasks[task_id]["status"] = "running"
            result = func(*args, **kwargs)
            tasks[task_id] = {"status": "done", "result": result}
        except Exception as e:
            tasks[task_id] = {"status": "error", "error": str(e)}
        queue.task_done()

def get_task(task_id):
    return tasks.get(task_id, {"status": "not_found"})
