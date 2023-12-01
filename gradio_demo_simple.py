import argparse
import datetime
import json
import os
import time

from typing import Tuple
from collections.abc import Mapping

import gradio as gr
from gradio_pdf import PDF

import requests
from PIL import Image

from jinja2 import FileSystemLoader, Environment


from llava.conversation import (Conversation, default_conversation,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import build_logger


from reportgen.report_maker import ReportMaker
from reportgen.report_export import ReportExporter
from reportgen.utils import image2b64
from reportgen.config import Settings as AppSettings


def get_new_conversation():
    settings = AppSettings()
    conv = default_conversation.copy()
    conv.system = settings.client.system_prompt
    return conv

def get_user_prompts():
    def _fix_prompt_item(i):
        if isinstance(i, Mapping):
            return next(iter(i.items()))
        return i

    settings = AppSettings()
    return list(map(_fix_prompt_item, settings.client.user_prompts))


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}


MODEL_SORT_PRIORITY = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def change_name(model: str) -> Tuple:
    if 'gpt' in model:
        return ("GPT (OpenAI)", model)
    if 'llava' in model:
        return ("LLaVA (Open Source)", model)
    return model


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    models = []

    try:
        requests.post(args.controller_url + "/refresh_all_workers").raise_for_status()
        ret = requests.post(args.controller_url + "/list_models")
        ret.raise_for_status()
        models.extend(ret.json()["models"])
    except (requests.HTTPError, requests.ConnectionError):
        logger.exception("Failed to get model list from controller:")

    models.sort(key=lambda x: MODEL_SORT_PRIORITY.get(x, x))
    logger.info(f"Models: {models}")

    models = list(map(change_name, models))

    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.update(
                value=model, visible=True,
                show_label=True
            )

    state = get_new_conversation()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = get_new_conversation()

    dropdown_update = gr.update(
        choices=models,
        value=models[0][1] if len(models) > 0 else ""
    )

    return state, dropdown_update


title_markdown = """
# Accident caption demo
"""


block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def do_submit_image_prompt(model_name: str, state: Conversation, files: list, situation_prompt: str, progress=gr.Progress()):
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")
    img_captioned = []

    if situation_prompt is None or len(situation_prompt) == 0:
        situation_prompt = "An accident occurred"

    for img_file in progress.tqdm(files, desc="Producing caption"):
        state_copy = state.copy()
        img = Image.open(img_file)
        prompt_user_text = situation_prompt

        # User prompt
        if '<image>' not in prompt_user_text:
            # text = '<Image><image></Image>' + text
            prompt_user_text = prompt_user_text + '\n<image>'

        state_copy.append_message(state_copy.roles[0], (prompt_user_text, img, 'Default'))

        # AI prompt indicator
        state_copy.append_message(state_copy.roles[1], None)

        logger.info("Situation: %s", situation_prompt)

        prompt = state_copy.get_prompt()
        logger.info("Entire prompt: %s", prompt)

        chat_payload = {
            "model": model_name,
            "prompt": prompt,
            "images": state_copy.get_images(),
            "temperature": 0.7,
            "top_p": 0.7,
            "max_new_tokens": 1536,
            "stop": state_copy.sep if state_copy.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state_copy.sep2,
            "data": state_copy.dict()
        }

        # print("Payload:", chat_payload)
        final_output = None

        try:
            # Stream output
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers=headers, json=chat_payload,
                stream=True, timeout=120
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        final_output = data["text"][len(prompt):].strip()
                        # state.messages[-1][-1] = output + "▌"
                        # yield (state, state.to_gradio_chatbot())
                    else:
                        output = data["text"] + f" (error_code: {data['error_code']})"
                        raise Exception("Chat model failed with error: %s" % output)
                        # state.messages[-1][-1] = output
                        # yield (state, state.to_gradio_chatbot())
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            logger.exception("Error running prompt:")
            raise e

        logger.info("Prompt complete: %s", final_output)
        img_captioned.append((
            image2b64(img),
            final_output
        ))

    return state, img_captioned, None


def do_make_template(captioned_images) -> str:
    settings = AppSettings()
    report_settings = settings.customer.report

    return ReportMaker(
        template_env=Environment(loader=FileSystemLoader(searchpath=report_settings.templates_dir)),
        additional_static_kwargs={
            "logo_url": report_settings.logo_image_url,
        }
    ).from_template(
        report_settings.report_template_file,
        img_captions=captioned_images
    )


def do_export_report(report_text: str, progress=gr.Progress()) -> Tuple[str, str]:
    settings = AppSettings()
    report_settings = settings.customer.report

    tdqm_iter = progress.tqdm(range(1), desc="Generating PDF")
    report_file = ReportExporter(method=report_settings.export_method).make_pdf(report_text)
    tdqm_iter.update()

    return report_file, gr.update(visible=True, value=report_file)


def build_demo(embed_mode):
    settings = AppSettings()

    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        captioned_img = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        label='Model Backend',
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True
                    )

                with gr.Row(variant="compact"):
                    eg_select_prompt = gr.Dropdown(
                        get_user_prompts(),
                        label="Incident claim type",
                        allow_custom_value=True
                    )

        with gr.Row():
            with gr.Column(scale=3):
                img_upload = gr.File(label="Upload media", file_count="multiple", file_types=settings.upload.allowed_types, interactive=True)
                submit_direct_btn = gr.Button(value="Submit", variant="primary", interactive=False)
                imagebox = gr.Gallery(label="Preview", visible=False)

                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                out_pdf = PDF(label="Preview")
                download_file = gr.File(label="Download report", visible=False)

        with gr.Row():
            with gr.Accordion("Report Template", open=False):
                in_txt = gr.Code(language="html")
                with gr.Row(variant="compact"):
                    generate_pdf_btn = gr.Button(
                        value="Generate Now",
                        size="sm",
                        variant="primary",
                        scale=1
                    )

        url_params = gr.JSON(visible=False)


        # Register listeners

        # Image upload handler. Update visibility of preview, and enable/disable submit button
        img_upload.change(
            lambda files: (
                gr.update(visible=(files is not None and len(files) > 0), value=files),
                gr.update(interactive=(files is not None and len(files) > 0))
            ),
            img_upload,
            [imagebox, submit_direct_btn]
        )

        submit_direct_btn.click(
            do_submit_image_prompt,
            inputs=[model_selector, state, img_upload, eg_select_prompt],
            outputs=[state, captioned_img, out_pdf],
            show_progress="full"
        ).success(
            do_make_template,
            inputs=[captioned_img],
            outputs=[in_txt],
            show_progress="full"
        ).success(
            do_export_report,
            inputs=in_txt,
            outputs=[out_pdf, download_file],
            trigger_mode='always_last',
            show_progress='full'
        )

        # Only generate PDF from code
        generate_pdf_btn.click(
            do_export_report,
            inputs=in_txt,
            outputs=[out_pdf, download_file],
            trigger_mode='always_last',
            show_progress='full'
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    pre_args = None

    if os.path.exists("config.json"):
        pre_args = json.load(open("config.json"))["args"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args(args=pre_args)
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
