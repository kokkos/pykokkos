#!/bin/bash

/usr/sbin/sshd -D -p 2222 &
/opt/nvidia/nvidia_entrypoint.sh
