import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

from deepx.dms import CIFAR10DM
from deepx.nn import ResNet18
from deepx.tasks import Classification

dm = CIFAR10DM


# 学習済みモデルの読み込み（ダミーコード）
def load_model(checkpoint_path):
    model = ResNet18(num_classes=10, in_channels=3)
    model = Classification.load_from_checkpoint(checkpoint_path, model=model, num_classes=dm.CLASSES)
    model.eval()
    return model

# 画像の前処理
def preprocess(image):
    transform = dm.transform()
    image = transform(image).unsqueeze(0)
    return image

# 推論
def predict(model, image):
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# ウェブアプリの作成
def classify_image(model, image):
    image = preprocess(image)
    prediction = predict(model, image)
    class_names = dm.CLASSES
    return class_names[prediction]

def app():
    # チェックポイントパスを入力するためのテキストボックスを作成
    checkpoint_path = gr.inputs.Textbox(lines=1, label="Checkpoint Path")
    model = None

    def load_model_and_classify_image(checkpoint_path, image):
        nonlocal model
        if model is None:
            model = load_model(checkpoint_path)
        return classify_image(model, image)

    # 画像の表示と推論結果の表示
    image = gr.inputs.Image()
    label = gr.outputs.Textbox()
    interface = gr.Interface(fn=load_model_and_classify_image, inputs=[checkpoint_path, image], outputs=label, capture_session=True)
    return interface

if __name__ == "__main__":
    app_instance = app()
    app_instance.launch()
