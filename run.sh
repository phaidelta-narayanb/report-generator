#!/bin/bash

echo "Runner starting..."

parallel -j4 --line-buffer --tag ::: './openai_worker.sh' './gradio_app.sh' './controller_app.sh'
