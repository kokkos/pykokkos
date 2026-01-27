FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Optional ARG for username (only set if secret is provided)
ARG USERNAME=""
ENV username="${USERNAME}"

# Optional ARG for SSH port
ARG PORT="2222"
ENV SSH_PORT="${PORT}"

# Dependencies installation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential=12.9ubuntu3 \
    micro=2.0.9-1ubuntu0.22.04.3 \
    nano=6.2-1ubuntu0.1 \
    git=1:2.34.1-1ubuntu1.15 \
    openssh-server=1:8.9p1-3ubuntu0.13 \
    wget=1.21.2-2ubuntu1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && ssh-keygen -A

# Set shell to bash with pipefail for safer pipe operations
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create user using Docker secrets for sensitive data
RUN --mount=type=secret,id=username \
    --mount=type=secret,id=password \
    --mount=type=secret,id=ssh_key \
    if [ -f "/run/secrets/username" ] && [ -s "/run/secrets/username" ] && [ -f "/run/secrets/password" ] && [ -s "/run/secrets/password" ]; then \
        export NAME && \
        NAME="$(cat /run/secrets/username)" && \
        export PASSWORD && \
        PASSWORD="$(cat /run/secrets/password)" && \
        useradd -m -u 1000 -s /bin/bash "${NAME}" && \
        echo "${NAME}:${PASSWORD}" | chpasswd && \
        echo "root:${PASSWORD}" | chpasswd && \
        usermod -aG sudo "${NAME}" && \
        chown -R "${NAME}":"${NAME}" "/home/${NAME}" && \
        echo "${NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
        echo "${NAME}" > /tmp/username; \
    else \
        echo "root" > /tmp/username; \
    fi

# Configure SSH permissions for devcontainer (only if custom user was created)
RUN export USERNAME && \
    USERNAME="$(cat /tmp/username)" && \
    if [ "${USERNAME}" != "root" ]; then \
        mkdir -p /var/run/sshd && \
        sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
        sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config; \
    fi

# Conditionally add SSH public key if provided
RUN --mount=type=secret,id=ssh_key \
    export USERNAME && \
    USERNAME="$(cat /tmp/username)" && \
    if [ -f /run/secrets/ssh_key ] && [ -s /run/secrets/ssh_key ] && [ "${USERNAME}" != "root" ]; then \
        export SSH_PUB_KEY && \
        SSH_PUB_KEY="$(cat /run/secrets/ssh_key)" && \
        mkdir -p "/home/${USERNAME}/.ssh/" && \
        echo "${SSH_PUB_KEY}" >> "/home/${USERNAME}/.ssh/authorized_keys" && \
        chmod 700 "/home/${USERNAME}/.ssh/" && \
        chown -R "${USERNAME}":"${USERNAME}" "/home/${USERNAME}/.ssh/" && \
        chmod 600 "/home/${USERNAME}/.ssh/authorized_keys"; \
    fi

# Set up custom entrypoint and init scripts
COPY dev-container/scripts/custom-entrypoint.sh /opt/custom-entrypoint.sh
COPY dev-container/scripts/init-pykokkos.sh /opt/init-pykokkos.sh
RUN chmod +x /opt/custom-entrypoint.sh /opt/init-pykokkos.sh

# Set working directory based on USERNAME ARG
# WORKDIR_PATH will be passed as build arg from build script
ARG WORKDIR_PATH="/root"

# Set WORKDIR - use WORKDIR_PATH from build arg
WORKDIR ${WORKDIR_PATH}

# Temporarily switch to user for conda and git operations
USER ${USERNAME:-root}

# Set up and initialize conda environment
# Install conda in the user's home directory
RUN export USERNAME && \
    USERNAME="$(cat /tmp/username)" && \
    if [ "${USERNAME}" != "root" ]; then \
        export HOME_DIR="/home/${USERNAME}" && \
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
        bash miniconda.sh -b -p "${HOME_DIR}/miniconda3" && \
        rm -f miniconda.sh && \
        echo "${HOME_DIR}/miniconda3/bin" > /tmp/conda_path; \
    else \
        export HOME_DIR="/root" && \
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
        bash miniconda.sh -b -p "${HOME_DIR}/miniconda3" && \
        rm -f miniconda.sh && \
        echo "${HOME_DIR}/miniconda3/bin" > /tmp/conda_path; \
    fi

# Initialize user's conda environment, TOS agreements and ensure it's loaded in every shell
RUN export CONDA_PATH && \
    CONDA_PATH="$(cat /tmp/conda_path)" && \
    "${CONDA_PATH}"/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    "${CONDA_PATH}"/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    "${CONDA_PATH}"/conda init

# Clone pykokkos repository (installation will happen at runtime)
RUN git clone https://github.com/kokkos/pykokkos.git

# Add conda to PATH in .bashrc
RUN export CONDA_PATH && \
    CONDA_PATH="$(cat /tmp/conda_path)" && \
    export HOME_DIR && \
    HOME_DIR="$(dirname "$(dirname "${CONDA_PATH}")")" && \
    if [ "${HOME_DIR}" != "/root" ]; then \
        echo "export PATH=\"${CONDA_PATH}:${HOME_DIR}/miniconda3/condabin:${HOME_DIR}/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" >> "${HOME_DIR}/.bashrc" && \
        echo "export condaPath=${CONDA_PATH}" >> "${HOME_DIR}/.bashrc"; \
    else \
        echo "export PATH=\"${CONDA_PATH}:${HOME_DIR}/miniconda3/condabin:${HOME_DIR}/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" >> "${HOME_DIR}/.bashrc" && \
        echo "export condaPath=${CONDA_PATH}" >> "${HOME_DIR}/.bashrc"; \
    fi

# Switch back to root for entrypoint execution (required for sshd and init scripts)
USER root

# Expose port in order to be able to connect from IDEs
EXPOSE ${SSH_PORT}

ENTRYPOINT ["/opt/custom-entrypoint.sh"]
