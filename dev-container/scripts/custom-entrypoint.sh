#!/bin/bash

# Get the username from environment (userName is set in Dockerfile)
USER_NAME=${userName:-pykokkos-dev}

# Install pykokkos-base on first run (when GPU is available) as the user
echo "hi from entrypoint!"
echo "Running as USER_NAME: ${USER_NAME}"
if [ -f /opt/pykokkos-install.sh ]; then
    echo "Found install script, executing as ${USER_NAME}..."
    su - ${USER_NAME} -c "/opt/pykokkos-install.sh"
else
    echo "Install script not found at /opt/pykokkos-install.sh"
fi

/usr/sbin/sshd -D -p 2222 &
/opt/nvidia/nvidia_entrypoint.sh
