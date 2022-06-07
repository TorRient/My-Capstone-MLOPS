FROM python:3.7

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -U pip

RUN pip install -r requirements.txt

RUN pip install protobuf~=3.19.0

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY . /app

EXPOSE 8080

CMD ["python", "deploy_api.py"]