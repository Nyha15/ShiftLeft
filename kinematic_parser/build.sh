#!/bin/bash
set -e

DOCKER_IMAGE_NAME="llm_robot"

echo "[*] Building Docker image: $DOCKER_IMAGE_NAME"
docker build -t $DOCKER_IMAGE_NAME .
