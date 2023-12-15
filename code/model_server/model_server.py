import os
import socket
import subprocess
from datetime import datetime

import redis

from scripts.svd import generate_video

import boto3
from botocore.client import Config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, BackgroundTasks

from contextlib import asynccontextmanager


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")


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


app = FastAPI(title="Model Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_redis():
    # Here, we re-use our connection pool
    # not creating a new one
    return redis.Redis(connection_pool=redis_pool)


def trainbg(model_id: str, prompt: str, max_steps: int, task_id: str):
    conn = get_redis()
    train_tasks = {"status": "TRAINING", "PROGRESS": "init"}
    conn.hmset(task_id, train_tasks)

    input_dir = f"images/{task_id}"
    # os.makedirs(task_id, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"input/{task_id}")
    if "Contents" in objects:
        for obj in objects["Contents"]:
            # Get the file key
            file_key = obj["Key"]

            # Extract file name from object key
            _, filename = os.path.split(file_key)

            # Download the file from S3
            if filename:  # Ensure the filename is not empty (for handling directories)
                s3_client.download_file(
                    bucket_name, file_key, f"images/{task_id}/{filename}"
                )
                # print(f"File {file_key} downloaded as {filename}")

    try:
        command = [
            "accelerate",
            "launch",
            "./scripts/train.py",
            "dreambooth",
            "--input-images-dir",
            input_dir,
            "--instance-prompt",
            prompt,
            "--resolution",
            "512",
            "--train-batch-size",
            "1",
            "--max-train-steps",
            str(max_steps),
            "--mixed-precision",
            "fp16",
            "--output-dir",
            f"./models/{model_id}",
            "--model-id",
            model_id,
            "--task-id",
            task_id,
        ]

        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        train_tasks = {"status": "MODEL TRAINED; UPLOADING"}
        conn.hmset(task_id, train_tasks)

        LOCAL_FILE_PATH = f"./models/{model_id}/pytorch_lora_weights.safetensors"
        with open(LOCAL_FILE_PATH, "rb") as file:
            s3_client.upload_fileobj(
                file, bucket_name, f"models/{model_id}/pytorch_lora_weights.safetensors"
            )

        train_tasks = {"status": "SUCCESS"}
        conn.hmset(task_id, train_tasks)
    except Exception as e:
        print(f"ERROR :: {e}")
        # train_tasks[model_id] = {"status": "ERROR"}
        train_tasks = {"status": "ERROR"}
        conn.hmset(task_id, train_tasks)
    server = {"status": "READY"}
    conn.hmset("server", server)


@app.post("/train_model")
async def train_model(
    prompt: str,
    model_id: str,
    max_steps: int,
    background_tasks: BackgroundTasks,
    task_id: str,
):
    conn = get_redis()
    server_status = conn.hgetall("server")
    if server_status:
        print(server_status)
        if server_status["status"] == "BUSY":
            return {
                "task-id": task_id,
                "message": f"training job failed: Reason: {server_status['reason']}",
            }

    server = {"status": "BUSY", "reason": "TRAINING"}
    conn.hmset("server", server)

    background_tasks.add_task(trainbg, model_id, prompt, max_steps, task_id)
    train_tasks = {"status": "PENDING"}
    conn.hmset(task_id, train_tasks)
    return {"task-id": task_id, "message": "training job submitted successfully"}


def generatebg(prompt: str, allow_motion: str, task_id: str, model_id: str):
    conn = get_redis()
    infer_tasks = {"status": "GENERATING IMAGES"}
    conn.hmset(task_id, infer_tasks)

    output_dir = f"output/{task_id}"
    output_model_dir = f"models/{task_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    try:
        command = [
            "python",
            "./scripts/train.py",
            "infer",
            "--prompt",
            prompt,
            "--output-dir",
            output_dir,
        ]

        if model_id:
            # download lora weight from s3 here
            s3_client.download_file(
                bucket_name,
                f"models/{model_id}/pytorch_lora_weights.safetensors",  # s3 url
                f"models/{task_id}/pytorch_lora_weights.safetensors",  # local path
            )
            command1 = [
                "--lora-weights",
                f"./models/{task_id}",
            ]
            command.extend(command1)

        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        infer_tasks = {"status": "UPLOADING IMAGES"}
        conn.hmset(task_id, infer_tasks)

        # upload image and text to s3
        for root, dirs, files in os.walk(output_dir):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                # s3dir = output_dir.split("/")[-2]
                s3_key = f"output/{task_id}/{file_name}"
                s3_client.upload_file(local_file_path, bucket_name, s3_key)

        if allow_motion == 'True':
            infer_tasks = {"status": "GENERATING VIDEO"}
            conn.hmset(task_id, infer_tasks)
            generate_video(f"{output_dir}/out_0.png", f"{output_dir}/generated.mp4")

            infer_tasks = {"status": "UPLOADING VIDEO"}
            conn.hmset(task_id, infer_tasks)

            s3_client.upload_file(
                f"{output_dir}/generated.mp4",
                bucket_name,
                f"output/{task_id}/generated.mp4",
            )

        infer_tasks = {"status": "GENERATION COMPLETE"}
        conn.hmset(task_id, infer_tasks)

    except Exception as e:
        print(f"ERROR :: {e}")
        infer_tasks = {"status": "ERROR"}
        conn.hmset(task_id, infer_tasks)

    server = {"status": "READY"}
    conn.hmset("server", server)


@app.post("/generate_model")
async def generate_model(
    prompt: str,
    allow_motion: str,
    background_tasks: BackgroundTasks,
    task_id: str,
    model_id: str = None,
):
    conn = get_redis()
    server_status = conn.hgetall("server")

    if server_status:
        print(server_status)
        if server_status["status"] == "BUSY":
            return {
                "task-id": task_id,
                "message": f"inference job failed: Reason: {server_status['reason']}",
            }
    server = {"status": "BUSY", "reason": "GENERATING"}
    conn.hmset("server", server)

    # current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    background_tasks.add_task(generatebg, prompt, allow_motion, task_id, model_id)
    infer_tasks = {"status": "PENDING"}
    conn.hmset(task_id, infer_tasks)
    return {"task-id": task_id, "message": "inference job submitted successfully"}


@app.get("/health")
async def health():
    return {"message": "ok"}


@app.get("/")
async def root():
    return {
        "message": f"Welcome to EMLO-S25 Model Server @ {socket.gethostbyname(socket.gethostname())}"
    }


# uvicorn model_server:app --host 0.0.0.0 --port 8080 --reload
