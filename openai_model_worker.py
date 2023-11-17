"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
from logging import getLogger
import time
import threading
from typing import Generator, Any
import uuid

import openai

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import uvicorn
from functools import partial


WORKER_HEART_BEAT_INTERVAL = 15
SERVER_ERROR_MESSAGE = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = getLogger(__name__)
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self, controller_addr, worker_addr, worker_id, no_register, model_name
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {repr(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + (
                    len(model_semaphore._waiters)
                    if model_semaphore._waiters is not None
                    else 0
                )
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @staticmethod
    def openai_completion_response_streaming(
        orig_prompt, **kwargs
    ) -> Generator[bytes, Any, None]:
        generated_text = orig_prompt
        streamer = openai.chat.completions.create(stream=True, **kwargs)
        for new_chunk in streamer:
            if len(new_chunk.choices) > 0:
                c0 = new_chunk.choices[0]
                if c0.delta.content is not None:
                    new_text = c0.delta.content
                    generated_text += new_text

                    yield json.dumps(
                        {"text": generated_text, "error_code": 0}
                    ).encode() + b"\0"

    def generate_stream(self, params) -> Generator[bytes, Any, None]:
        gpt_args = {
            "model": params["model"],
            "messages": [],
            "max_tokens": params["max_new_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "stop": params["stop"],
        }
        messages = gpt_args["messages"]

        if params["data"]["system"] is not None and len(params["data"]["system"]) > 0:
            messages.append({"role": "system", "content": params["data"]["system"]})

        for msg in params["data"]["messages"]:
            new_msg = {}
            new_msg["role"] = msg[0].lower()
            new_msg["content"] = []

            if msg[1] is not None:
                new_msg["content"].append({"type": "text", "text": msg[1]})

                if "<image>" in msg[1]:
                    new_msg["content"].append(
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,"
                            + params["images"][0],
                        }
                    )

            if len(new_msg["content"]) > 0:
                messages.append(new_msg)

        yield from self.openai_completion_response_streaming(
            params["prompt"], **gpt_args
        )

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": SERVER_ERROR_MESSAGE,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": SERVER_ERROR_MESSAGE,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        partial(release_model_semaphore, fn=worker.send_heart_beat)
    )
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-name", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_name,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
