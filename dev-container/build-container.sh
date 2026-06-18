#!/bin/bash

# Get absolute paths for all directories we need
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
CONTAINER_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(dirname "$CONTAINER_DIR")
SECRETS_FILE="${CONTAINER_DIR}/secrets.yaml"
SECRETS_TEMPLATE="${CONTAINER_DIR}/secrets-template.yaml"

# Check if secrets.yaml exists, if not copy from template
if [ ! -f "${SECRETS_FILE}" ]; then
    if [ -f "${SECRETS_TEMPLATE}" ]; then
        echo "secrets.yaml not found. Creating from template..."
        cp "${SECRETS_TEMPLATE}" "${SECRETS_FILE}"
        echo ""
        echo "======================================"
        echo "CONFIGURATION (OPTIONAL)"
        echo "======================================"
        echo "A secrets.yaml file has been created from the template."
        echo "You can edit it with your configuration or skip this step:"
        echo ""
        echo "  ${SECRETS_FILE}"
        echo ""
        echo "Run this script again to build the container."
        echo "======================================"
        exit 0
    else
        echo "Error: Neither secrets.yaml nor secrets-template.yaml found!"
        exit 1
    fi
fi

# Template values
TEMPLATE_USERNAME="root"
TEMPLATE_PASSWORD="pykPassword"
TEMPLATE_PORT=""
TEMPLATE_SSH=""

# Function to parse YAML value (simple key: value parser)
parse_yaml_value() {
    local key="$1"
    local file="$2"
    # Remove quotes and extract value after colon
    grep "^${key}:" "$file" | sed "s/^${key}:[[:space:]]*//" | sed 's/^"//' | sed 's/"$//' | sed "s/^'//" | sed "s/'$//"
}

# Function to check if a value is not a template (i.e., should be used)
# Modifies the variable: keeps value if not template, clears if template
# Returns true/false
is_not_template() {
    local var_name="$1"
    local template="$2"
    local value="${!var_name}"

    if [ "${value}" != "${template}" ] && [ -n "${value}" ]; then
        # Variable is not a template, keep its value
        echo "true"
    else
        # Variable is a template, clear it
        eval "${var_name}=\"\""
        echo "false"
    fi
}

# Load secrets from YAML file
load_secrets() {
    CONTAINER_NAME=$(parse_yaml_value "container_name" "${SECRETS_FILE}")
    CONTAINER_NAME=${CONTAINER_NAME:-"pyk"}
    USERNAME=$(parse_yaml_value "username" "${SECRETS_FILE}")
    PASSWORD=$(parse_yaml_value "password" "${SECRETS_FILE}")
    PORT=$(parse_yaml_value "port" "${SECRETS_FILE}")
    SSH_KEY=$(parse_yaml_value "ssh" "${SECRETS_FILE}")
}

# Check which values are templates
check_templates() {
    HAS_USERNAME=$(is_not_template USERNAME "${TEMPLATE_USERNAME}")
    HAS_PASSWORD=$(is_not_template PASSWORD "${TEMPLATE_PASSWORD}")
    HAS_SSH=$(is_not_template SSH_KEY "${TEMPLATE_SSH}")
    HAS_PORT=$(is_not_template PORT "${TEMPLATE_PORT}")
}

# Build Docker build arguments
build_docker_args() {
    BUILD_ARGS=()
    SECRET_FILES=()

    # Create temporary directory for secrets
    SECRETS_DIR=$(mktemp -d)

    if [ "$HAS_USERNAME" = "true" ]; then
        BUILD_ARGS+=(--build-arg "USERNAME=${USERNAME}")
        BUILD_ARGS+=(--build-arg "WORKDIR_PATH=/home/${USERNAME}")
        echo -n "${USERNAME}" > "${SECRETS_DIR}/username"
        SECRET_FILES+=("--secret" "id=username,src=${SECRETS_DIR}/username")
    else
        BUILD_ARGS+=(--build-arg "WORKDIR_PATH=/root")
        # Create empty secret file to avoid errors
        touch "${SECRETS_DIR}/username"
        SECRET_FILES+=("--secret" "id=username,src=${SECRETS_DIR}/username")
    fi

    if [ "$HAS_PASSWORD" = "true" ]; then
        echo -n "${PASSWORD}" > "${SECRETS_DIR}/password"
        SECRET_FILES+=("--secret" "id=password,src=${SECRETS_DIR}/password")
    else
        # Create empty secret file to avoid errors
        touch "${SECRETS_DIR}/password"
        SECRET_FILES+=("--secret" "id=password,src=${SECRETS_DIR}/password")
    fi

    if [ "$HAS_SSH" = "true" ]; then
        echo -n "${SSH_KEY}" > "${SECRETS_DIR}/ssh_key"
        SECRET_FILES+=("--secret" "id=ssh_key,src=${SECRETS_DIR}/ssh_key")
    else
        # Create empty secret file to avoid errors
        touch "${SECRETS_DIR}/ssh_key"
        SECRET_FILES+=("--secret" "id=ssh_key,src=${SECRETS_DIR}/ssh_key")
    fi

    [ "$HAS_PORT" = "true" ] && BUILD_ARGS+=(--build-arg "PORT=${PORT}")
}

# Build Docker container
build_container() {
    echo "Building docker container: ${CONTAINER_NAME}"
    # TODO: add `--no-cache` after confirming that script can be builded
    docker build \
        "${BUILD_ARGS[@]}" \
        "${SECRET_FILES[@]}" \
        -t "${CONTAINER_NAME}:latest" \
        -f "${CONTAINER_DIR}/Dockerfile" \
        "${PROJECT_ROOT}"

    local build_result=$?

    # Cleanup secrets directory
    if [ -n "${SECRETS_DIR}" ] && [ -d "${SECRETS_DIR}" ]; then
        rm -rf "${SECRETS_DIR}"
    fi

    if [ $build_result != 0 ]; then
        echo "Docker build failed"
        exit 1
    fi
}

# Stop and remove existing container
stop_and_remove_container() {
    echo "Stopping ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true

    echo "Removing ${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
}

# Run Docker container
run_container() {
    RUN_ARGS=(
        -dit
        --gpus all
        --security-opt seccomp=unconfined
        --name "${CONTAINER_NAME}"
        --restart unless-stopped
    )

    [ "$HAS_PORT" = "true" ] && RUN_ARGS+=(-p "${PORT}:${PORT}")
    RUN_ARGS+=("${CONTAINER_NAME}")
    # Explicitly pass bash to ensure container keeps running with entrypoint
    RUN_ARGS+=("bash")

    docker run "${RUN_ARGS[@]}"
}

# Display farewell message
farewell() {
    EXEC_USERNAME=$([ "$HAS_USERNAME" = "true" ] && echo "${USERNAME}" || echo "root")
    echo "##################"
    echo "Container started."
    echo "To access container from terminal, enter: "
    echo "> docker exec -it -u ${EXEC_USERNAME} ${CONTAINER_NAME} bash"
    [ "$HAS_PASSWORD" = "true" ] && echo "To access sudo mode, enter 'su' and password from secrets.yaml."
    echo ""
    echo "PyKokkos installation is running in the background."
    echo "To check installation progress, run:"
    echo "> docker exec -u root ${CONTAINER_NAME} bash -c \"cat /tmp/pyk-install.log\""
    echo "##################"
}

# Main script execution flow
pushd "$CONTAINER_DIR" &> /dev/null || exit

load_secrets
check_templates
build_docker_args
build_container
stop_and_remove_container
run_container
farewell

popd || exit
