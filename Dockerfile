FROM emacski/tensorflow-serving:latest

COPY ./serving_model_dir /models
ENV MODEL_NAME=emotion-classification-model