#!/bin/bash

# Get absolute paths for all directories we need
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
CONTAINER_DIR=$(dirname "$SCRIPT_PATH")
SCRIPTS_DIR="${CONTAINER_DIR}/scripts"
PROJECT_ROOT=$(dirname "$CONTAINER_DIR")

pushd "$CONTAINER_DIR" &> /dev/null

# Secrets initialization
bash "${SCRIPTS_DIR}/init.sh"

# Container name initialization
DEFAULT_CONTAINER_NAME="pykokkos-dev-container"
echo "Container name configuration. Press 'Enter' to use default"
read -p "Enter container name [${DEFAULT_CONTAINER_NAME}]: " CONTAINER_NAME
CONTAINER_NAME=${CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}
USERNAME="$(cat ${CONTAINER_DIR}/secrets/username)"

# Docker container build (add secret to use in `RUN` commands and arg to use in 
# Dockerfile commands)
echo "Building docker container: ${CONTAINER_NAME}"

docker build \
    --no-cache \
    --build-arg USERNAME="${USERNAME}" \
    --secret id=username,src="${CONTAINER_DIR}/secrets/username" \
    --secret id=password,src="${CONTAINER_DIR}/secrets/pass" \
    --secret id=ssh_key,src="${CONTAINER_DIR}/secrets/sha" \
    -t "${CONTAINER_NAME}:latest" \
    -f "${CONTAINER_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"

if [ $? != 0 ]; then 
    exit
fi
# Stopping same environment, if exists
echo "Stopping ${CONTAINER_NAME}"
docker stop "${CONTAINER_NAME}"

echo "Removing ${CONTAINER_NAME}"
docker rm "${CONTAINER_NAME}"

docker run -dit \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    -p 2222:2222 \
    --restart unless-stopped \
    "${CONTAINER_NAME}"

function farewell {
    echo "##################"
    echo "Container started."
    echo "To access container from terminal, enter: "
    echo "> docker exec -it -u ${USERNAME} ${CONTAINER_NAME} bash"
    echo "To access sudo mode, enter 'su' and password, entered/generated above."
}

# Call farewell function at the end
farewell
popd
