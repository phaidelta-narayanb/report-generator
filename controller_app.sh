#!/bin/bash

echo "Starting controller..."

python3 -m llava.serve.controller --host 0.0.0.0 --port 10000
