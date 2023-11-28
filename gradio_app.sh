#!/bin/bash

echo "Starting Gradio..."

until python3 gradio_demo_simple.py --controller "http://localhost:10000" --model-list-mode reload; do
    echo "Gradio server crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
