import os
import zlib
import socket
from datetime import datetime

import redis
import httpx
import requests

import boto3
from botocore.client import Config


from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8080")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global redis_pool, s3_client, bucket_name
    print(f"creating redis connection with {REDIS_HOST=} {REDIS_PORT=}")
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True,
    )
    s3_client = boto3.client(
        "s3",
        config=Config(
            region_name="us-west-2",
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )
    bucket_name = "emlos325c"
    yield
    # Clean up the ML models and release the resources
    del redis_pool


app = FastAPI(title="Web Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_redis():
    return redis.Redis(connection_pool=redis_pool)


@app.post("/train")
async def train(
    prompt: str,
    model_id: str,
    files: list[UploadFile],
    max_steps: int,
):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    input_dir = f"./data/{current_datetime}/"
    os.makedirs(input_dir, exist_ok=True)  # Create folder if it doesn't exist
    task_id = f"{current_datetime}"

    for file in files:
        contents = await file.read()
        s3_key = f"input/{task_id}/{file.filename}"
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=contents)

    try:
        url = f"{MODEL_SERVER_URL}/train_model"
        query_param = {
            "prompt": prompt,
            "model_id": model_id,
            "max_steps": max_steps,
            "task_id": task_id,
        }
        response = requests.post(url, params=query_param)
        return response.text

    except Exception as e:
        print(f"ERROR :: {e}")
        raise HTTPException(status_code=500, detail="Error from Model Endpoint")


@app.post("/generate/lora")
async def generate(
    prompt: str,
    allow_motion: bool,
    model_id: str = None,
):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    task_id = f"{current_datetime}"

    try:
        url = f"{MODEL_SERVER_URL}/generate_model"
        query_param = {
            "prompt": prompt,
            "model_id": model_id,
            "allow_motion": allow_motion,
            "task_id": task_id,
        }
        response = requests.post(url, params=query_param)
        return response.text

    except Exception as e:
        print(f"ERROR :: {e}")
        raise HTTPException(status_code=500, detail="Error from Model Endpoint")


@app.post("/generate/sdxl")
async def generatesdxl(
    prompt: str,
    allow_motion: bool,
):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    task_id = f"{current_datetime}"

    try:
        url = f"{MODEL_SERVER_URL}/generate_model"
        query_param = {
            "prompt": prompt,
            "allow_motion": allow_motion,
            "task_id": task_id,
        }
        response = requests.post(url, params=query_param)
        return response.text

    except Exception as e:
        print(f"ERROR :: {e}")
        raise HTTPException(status_code=500, detail="Error from Model Endpoint")


@app.get("/result")
async def train_result(
    task_id: str,
):
    conn = get_redis()
    train_status = conn.hgetall(task_id)

    if train_status is None:
        return {"ERROR": "incorrect task id"}
    if train_status["status"] == "TRAINING":
        return {"status": "TRAINING", "progress": train_status["PROGRESS"]}
    return {"status": train_status["status"]}


@app.get("/images")
async def gen_result(
    task_id: str,
):
    conn = get_redis()
    gen_status = conn.hgetall(task_id)

    if gen_status is None:
        return {"ERROR": "incorrect task id"}
    if gen_status["status"] == "GENERATION COMPLETE":
        obj_prefix = f"output/{task_id}/generated.mp4"
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=obj_prefix)

        if "Contents" not in response:
            # .mp4 does not exist
            obj_prefix = f"output/{task_id}/out_grid.png"

        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": obj_prefix},
            ExpiresIn=3600 * 1,  # 1 hour
        )
        return {"status": "MEDIA READY", "url": presigned_url}
    return {"status": gen_status["status"]}


@app.get("/")
async def root():
    return {
        "message": f"Welcome to EMLO-S25 Web Server @ {socket.gethostbyname(socket.gethostname())}"
    }


@app.get("/health")
async def health():
    return {"message": "ok"}


# uvicorn web_server:app --host 0.0.0.0 --port 9080 --reload
