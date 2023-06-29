import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image, ImageTk

from vision.nn import available_models
from vision.tasks.classification import ClassificationTask


class DeepLearningApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Deep Learning Inference App")
        self.window.geometry("400x400")

        self.ckpt_path = ""
        self.image_path = ""

        self.model_name_label = tk.Label(window, text="Model:", font=("Helvetica", 12, "bold"))
        self.model_name_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self.dataset_name_label = tk.Label(window, text="Dataset:", font=("Helvetica", 12, "bold"))
        self.dataset_name_label.grid(row=1, column=0, sticky="w", padx=10)

        image_frame = tk.Frame(window, bg="#ECECEC", width=300, height=300)
        image_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        self.result_label = tk.Label(
            window, text="Inference result", font=("Helvetica", 12, "bold")
        )
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

        ckpt_button = tk.Button(
            window, text="Select checkpoint", command=self.select_ckpt_path, font=("Helvetica", 12)
        )
        ckpt_button.grid(row=4, column=0, pady=10)

        image_button = tk.Button(
            window, text="Select image", command=self.select_image, font=("Helvetica", 12)
        )
        image_button.grid(row=4, column=1, pady=10)

        inference_button = tk.Button(
            window, text="Run inference", command=self.run_inference, font=("Helvetica", 14, "bold")
        )
        inference_button.grid(row=5, column=0, columnspan=2, pady=10)

    def select_ckpt_path(self):
        self.ckpt_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch Model Files", "*.ckpt"), ("All Files", "*.*")),
            initialdir="../../experiments",
        )
        if self.ckpt_path:
            self.model_name, self.dataset_name = self.get_model_dataset_name(self.ckpt_path)
            self.model_name_label.config(text="Model: " + self.model_name)
            self.dataset_name_label.config(text="Dataset: " + self.dataset_name)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image Files", "*.jpg"), ("All Files", "*.*")),
            initialdir="../../data",
        )
        if self.image_path:
            print("Selected image file:", self.image_path)
            image = Image.open(self.image_path)
            image.thumbnail((300, 300))
            self.image = image
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def run_inference(self):
        print("Running inference...")
        if self.ckpt_path and self.image_path:
            model = ClassificationTask.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                data_dir="",
            ).to("cuda")
            # img = self.image.resize(model.dataset["size"])
            # img = np.array(img)
            # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
            img = model.dataset["transform"](self.image).unsqueeze(0).to("cuda")
            # print(img.shape)

            model.eval()
            with torch.no_grad():
                result = model.predict_step(img, None)
            result = model.dataset["classes"][result[0].item()]

            self.result_label.configure(text="Prediction: " + result)
        else:
            print(self.ckpt_path)
            print(self.image_path)
            print("Model file or image file not selected.")

    def get_model_dataset_name(self, ckpt_path):
        model = None
        for m in available_models:
            if m in ckpt_path:
                model = m
                break
        try:
            dataset = ckpt_path.split(model)[0].split("/")[-2]
        except IndexError:
            raise ValueError("Dataset name not found in checkpoint path.")
        return model, dataset


if __name__ == "__main__":
    window = tk.Tk()
    app = DeepLearningApp(window)
    window.mainloop()
