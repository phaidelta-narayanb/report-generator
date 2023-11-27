#!/bin/bash

echo "Starting OpenAI worker..."

until python3 openai_model_worker.py --controller-address "http://localhost:10000" --host 0.0.0.0 --model-name gpt-4-vision-preview; do
    echo "OpenAI worker crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
