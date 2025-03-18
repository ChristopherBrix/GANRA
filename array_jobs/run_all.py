import time
import os

TASKS_LIST = "tasks.txt"
TASKS_LIST_LOCK = "tasks.lock"
TASKS_LIST_LOCK_FREE = "tasks.lock.free"

sleep = False
while True:
    if sleep:
        time.sleep(5)
        sleep = False
    command = None

    if os.path.exists(TASKS_LIST_LOCK):
        print(f"[{time.time()}] Lock file exists", flush=True)
        sleep = True
        continue

    try:
        os.rename(TASKS_LIST_LOCK_FREE, TASKS_LIST_LOCK)
    except FileNotFoundError:
        print(f"[{time.time()}] Failed to aquire lock", flush=True)
        sleep = True
        continue

    with open(TASKS_LIST, "r") as f:
        lines = f.readlines()
    if len(lines) == 0:
        print(f"[{time.time()}] File {TASKS_LIST} is empty, shutting down", flush=True)
        os.rename(TASKS_LIST_LOCK, TASKS_LIST_LOCK_FREE)  # freeing file before breaking
        break
    else:
        with open(TASKS_LIST, "w") as f:
            f.writelines(lines[1:])
        command = lines[0]

    os.rename(
        TASKS_LIST_LOCK, TASKS_LIST_LOCK_FREE
    )  # give signal to all other jobs that file can now be accessed

    if command is not None:
        print(f"[{time.time()}] Will run `{lines[0]}`", flush=True)
        os.system(lines[0])
