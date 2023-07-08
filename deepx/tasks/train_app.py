import argparse

import gradio as gr


def create_manager(task, dataset, **kwargs):
    if task == "classification":
        from deepx.tasks.vision.classification import Classification

        task = Classification(dataset_name=dataset, **kwargs)
    # elif task == "detection":
    #     from deepx.tasks.vision.detection import DetectionTask

    #     task = DetectionTask(dataset_name=dataset, **kwargs)
    elif task == "segmentation":
        from deepx.tasks.vision.segmentation import Segmentation

        task = Segmentation(dataset_name=dataset, **kwargs)

    elif task == "langmodel":
        from deepx.tasks.language.langmodel import LangModeling

        task = LangModeling(dataset_name=dataset, **kwargs)


def change_radio(choice):
    if choice == "classification":
        return gr.update(visible=True, value="Classification")
    elif choice == "detection":
        return gr.update(visible=True, value="Detection")
    elif choice == "segmentation":
        return gr.update(visible=True, value="Segmentation")
    else:
        return gr.update(visible=False)


with gr.Blocks() as app:
    with gr.Tab("Vision"):
        task = gr.Radio(["Classification", "Detection", "Segmentation"], label="Task")

    with gr.Tab("Language"):
        task = gr.Radio(["Translation", "Language model"], label="Task")

    gr.Button("Start Training").click(
        fn=create_manager, inputs=[model, dataset, epochs, batch_size]
    )
