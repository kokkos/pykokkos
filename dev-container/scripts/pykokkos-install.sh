#!/bin/bash

USERNAME=$(whoami)
if [ ! -f /home/"${USERNAME}"/.pyk-installed ]; then
    echo "PyKokkos is not installed yet"

    sleep 4
    touch /home/"${USERNAME}"/.pyk-installed
fi