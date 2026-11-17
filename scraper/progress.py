import json
import os
from datetime import datetime

PROGRESS_FILE = "../data/progress.json"


def save_progress(make_index, page_number, total_cars):

    os.makedirs("../data", exist_ok=True)

    data = {
        "make_index": make_index,
        "page_number": page_number,
        "total_cars": total_cars,
        "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_progress():

    if not os.path.exists(PROGRESS_FILE):
        return None

    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


def clear_progress():

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)