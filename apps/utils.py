from tkinter import filedialog


def set_ckpt_path():
    path = filedialog.askopenfilename(
        initialdir="/workspace/experiments/mlruns",
        title="Select checkpoint file",
    )
    return path
