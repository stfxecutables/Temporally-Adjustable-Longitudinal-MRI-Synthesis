from pathlib import Path
import os

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


if __name__ == "__main__":
    cur_folder = DATA_ROOT / "open_ms_data"

    for i in range(3, 21):
        if i < 10:
            tmp = "0" + str(i)
        else:
            tmp = str(i)
        print("mkdir " + tmp)
        new_folder = cur_folder / f"patient{tmp}"
        os.mkdir(new_folder)
