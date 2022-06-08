# 🧬 MLOps Pipeline - Scan Document - Task: Detect + OCR
[![model-deploy-on-release](https://github.com/MLOPsStudyGroup/dvc-gitactions/actions/workflows/deploy_on_release.yaml/badge.svg)](https://github.com/TorRient/My-Capstone-MLOPS/blob/master/.github/workflows/deploy_on_release.yaml)

## 🔰 Milestones
- [X] Data Versioning: DVC
- [X] Machine Learning Pipeline: DVC Pipeline (preprocess, train, evaluate)
- [X] Model Registry: DVC
- [O] Monitor model: Weights & Biases and MLflow
- [X] CI/CD: Github Actions
- [X] CML: Continuous Machine Learning and Github Actions
- [X] Deploy on release: Github Actions and Kubernetes on GCP
- [X] Rolling update: Kustomize - autoscale - auto progresive delivery - Kubernetes
- [O] Monitor: Grafana

## 📋 Requirements

* DVC
* Python3 and pip
* Access to Google Cloud Platform

## 🏃🏻 Running Project
### ⚗️ Using DVC

Download data from the DVC repository (analog to ```git pull```)
```
export GOOGLE_APPLICATION_CREDENTIALS="path to CREDENTIALS json file in GCP"
dvc pull
```

Reproduces the pipeline using DVC
```
dvc repro
```


### ⚙️ DVC Pipelines


✂️ Preprocessing pipeline
```
dvc run -n preprocess -d dataset_val -o dataset_val.txt \
python3 create_dataset.py
```


📘 Training pipeline
```
dvc run -n train -d ./src/tool/training.py -d ./dataset_train/ -d params.yaml -o ./weights/ocr_model.pth \
python3 ./src/tool/training.py --config params.yaml
```


📊 Evaluate pipeline
```
dvc run -n evaluate -d ./src/tool/evaluate.py -d ./dataset_val/ -d params.yaml \
python3 ./src/tool/evaluate.py --config params.yaml
```

### 🐙 Git Actions
🔐 Google Cloud Platform and Github Credentials

To use Git Actions to deploy your model, you'll need to encrypt it, to do that run the command bellow and choose a strong password.

Now in the GitHub page for the repository, go to ```Settings->Secrets``` and add the keys to the following secrets:

```
- GCP_PROJECT_ID: id project in your GCP
- GOOGLE_APPLICATION_CREDENTIALS_DATA: CREDENTIALS in your GCP
- PERSONAL_ACCESS_TOKEN: access token in your github
```