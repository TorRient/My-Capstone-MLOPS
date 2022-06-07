import os
import numpy as np
import yaml
import uvicorn
from PIL import Image
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

from PIL import Image

import sys
sys.path.append("../")

from tool.predictor import Predictor
from tool.config import Cfg

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

@app.post('/predict')
async def predict(image_c: UploadFile=File(...)):
    img = Image.open(image_c.file).convert('RGB')
    img = np.asarray(img)
    results = ocr.ocr(img, rec=False, cls=False)
    boxes = np.asarray([line for line in results])
    boxes = boxes[boxes[:, 0, 1].argsort()]
    img_list = []
    result = []
    box_r = []
    for i, box in enumerate(boxes):
        x_min = int(min(box[:, 0]))
        x_max = int(max(box[:, 0]))
        y_min = int(min(box[:, 1]))
        y_max = int(max(box[:, 1]))
        box_r.append([x_min, y_min, x_max, y_max])
        box_text = img[y_min:y_max, x_min:x_max]
        img_list.append(Image.fromarray(box_text))
        if (i + 1) % 16 == 0:
            sent = detector.predict_batch(img_list)
            img_list = []
            for j, s in enumerate(sent):
                result.append({"text": s, "box": box_r[j]})
            box_r = []
    if 0 < len(img_list) < 16:
        sent = detector.predict_batch(img_list)
        for j, s in enumerate(sent):
            result.append({"text": s, "box": box_r[j]})

    return {
        "result": result
    }

if __name__ == "__main__":
    if not os.path.exists("params.yaml"):
        os.system("cp ../params.yaml ./")
    params_yaml = "params.yaml"
    config_yaml = read_params(params_yaml)
    config = Cfg.load_config_from_file(config_yaml["model_name"].replace("src/", ""))
    detector = Predictor(config)
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8080)