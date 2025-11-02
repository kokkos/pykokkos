FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG USERNAME
ENV username=${USERNAME}

# Dependencies installation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    micro \
    nano \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Read secrets and create user with home directory
RUN --mount=type=secret,id=username \
    --mount=type=secret,id=password \
    --mount=type=secret,id=ssh_key \
    export NAME=$(cat /run/secrets/username) && \
    export PASSWORD=$(cat /run/secrets/password) && \
    export SSH_PUB_KEY=$(cat /run/secrets/ssh_key) && \
    useradd -m -u 1000 -s /bin/bash ${NAME} && \
    echo "${NAME}:${PASSWORD}" | chpasswd && \
    echo "root:${PASSWORD}" | chpasswd && \
    usermod -aG sudo ${NAME} && \
    chown -R ${NAME}:${NAME} /home/${NAME} && \
    echo "${NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "${NAME}" > /tmp/username

# Configure SSH permissions for devcontainer
RUN mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Add SSH public key if provided
RUN --mount=type=secret,id=ssh_key \
    SSH_PUB_KEY=$(cat /run/secrets/ssh_key) && \
    mkdir -p /home/${username}/.ssh/ && \
    if [ -n "$SSH_PUB_KEY" ]; then \
        echo "$SSH_PUB_KEY" >> /home/${username}/.ssh/authorized_keys; \
    fi && \
    chmod 700 /home/${username}/.ssh/ && \
    chown -R ${username}:${username} /home/${username}/.ssh/ && \
    if [ -f /home/${username}/.ssh/authorized_keys ]; then \
        chmod 600 /home/${username}/.ssh/authorized_keys; \
    fi

# Set up custom entrypoint
WORKDIR /home/${username}/
COPY dev-container/scripts/custom-entrypoint.sh /opt/custom-entrypoint.sh
RUN chmod +x /opt/custom-entrypoint.sh

# Set up and initialize conda environment
RUN echo "export condaPath=/home/${username}/miniconda3/bin/" >> /tmp/conda_env
ENV condaPath="/home/${username}/miniconda3/bin"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /home/${username}/miniconda3 && \
    rm -f miniconda.sh && \
    chown -R ${username}:${username} /home/${username}/miniconda3/

# Initialize user's conda environment, TOS agreements
USER ${username}
WORKDIR /home/${username}/
# Initialize conda in bash and ensure it's loaded in every shell
RUN ${condaPath}/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${condaPath}/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    ${condaPath}/conda init

# Clone myself to this container
RUN git clone https://github.com/kokkos/pykokkos.git

# Expose port in order to be able to connect from IDEs
EXPOSE 2222

# Add ~/.local/bin to PATH (for pip installs etc.)
RUN echo "export PATH=\"/home/${username}/miniconda3/bin:/home/${username}/miniconda3/condabin:/home/${username}/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" >> /home/${username}/.bashrc

ENTRYPOINT ["/opt/custom-entrypoint.sh"]