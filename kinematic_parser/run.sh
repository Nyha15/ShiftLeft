#!/bin/bash
set -e

DOCKER_IMAGE_NAME="llm_robot_task"
CONTAINER_NAME="llm_robot_container"

# Mount current directory into the container's /workspace
# Use --gpus all if NVIDIA driver ≥ 470 is installed on host
echo "[*] Running Docker container: $CONTAINER_NAME"

docker run -it --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    $DOCKER_IMAGE_NAME bash
