FROM python:3.10-slim

WORKDIR /app

COPY ./inference_requirements.txt /app

RUN pip3 install --upgrade --no-cache-dir -r ./inference_requirements.txt

COPY ./src /app/src

VOLUME ["/app/data", "/app/logs"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 