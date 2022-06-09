import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import gradio as gr
import uuid

def inference(img):
    img = Image.fromarray(img)
    mem_file_c = BytesIO()
    img.save(mem_file_c, "PNG", quality=100)
    mem_file_c.seek(0)
    all_line = ""
    r = requests.post(
                f"http://34.71.54.46/predict",
                files={
                    'image_c': ('image.PNG', mem_file_c, 'image/png'),
                }
            )
    if type(r.json()["result"]) == list:
        txts = [line["text"] for line in r.json()["result"]]
        boxes = np.asarray([line["box"] for line in r.json()["result"]])
        img = np.asarray(img)
        for i, box in enumerate(boxes):
            x_min = box[0]
            x_max = box[2]
            y_min = box[1]
            y_max = box[3]
            frontscale = (y_max-y_min)/(60)
            img = cv2.putText(img, str(i+1), (x_min-15, int((y_max+y_min)/2)), cv2.FONT_HERSHEY_SIMPLEX, frontscale, (0,0,255), 1, cv2.LINE_AA)
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        for i, txt in enumerate(txts):
            all_line += str(i+1) + ": " + txt + "\n"
        img = Image.fromarray(img)
    id = str(uuid.uuid4().hex)
    img.save(id + '_result.jpg')
    return f'{id}_result.jpg', all_line

title = 'UI - Detect line and OCR for Vietnamese document'
description = 'Gradio Detect line and OCR demo for Vietnamese document'
examples = ['vb.png', 'vb2.png', 'bk.JPG']
css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"

inputs = gr.inputs.Image()
o1 = gr.outputs.Image()
o2 = gr.outputs.Textbox()
gr.Interface(
    inference,
    inputs=inputs,
    outputs=[o1, o2],
    title=title,
    description=description,
    examples=examples,
    css=css,
    enable_queue=True
    ).launch(debug=True)