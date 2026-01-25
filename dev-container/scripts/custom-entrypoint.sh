#!/bin/bash

# Define helper function used by nvidia banner scripts
print_banner_text() {
    local banner_char=$1
    local text="$2"
    local pad="${banner_char}${banner_char}"
    local repeats=$((${#text} + 6))
    printf '%*s\n' "$repeats" '' | tr ' ' "$banner_char"
    echo "${pad} ${text} ${pad}"
    printf '%*s\n' "$repeats" '' | tr ' ' "$banner_char"
}

SSH_PORT=${SSH_PORT:-2222}
/usr/sbin/sshd -D -p" ${SSH_PORT}" &

# Source nvidia environment parts (without the exec at the end)
if [ -d /opt/nvidia/entrypoint.d ]; then
    # Process .txt files (display banners)
    for _file in /opt/nvidia/entrypoint.d/*.txt; do
        [ -f "$_file" ] && cat "$_file"
    done

    # Process .sh files (source environment setup)
    for _file in /opt/nvidia/entrypoint.d/*.sh; do
        # shellcheck source=/dev/null
        [ -f "$_file" ] && source "$_file"
    done
    echo
fi

# Run PyKokkos initialization script in background
if [ -f /opt/init-pykokkos.sh ]; then
    /opt/init-pykokkos.sh &
fi

# Keep container running - if no command provided, sleep forever
if [ $# -eq 0 ]; then
    exec tail -f /dev/null
else
    exec "$@"
fi
