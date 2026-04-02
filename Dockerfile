ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT=/app/cloud_sync/efficientnet_b2/best_model.pt

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements-docker.txt

COPY . /app

EXPOSE 9053

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9053"]
