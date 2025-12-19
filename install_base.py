#!/usr/bin/env python
"""
Wrapper script to install pykokkos-base from the root pykokkos directory.
This script allows installing pykokkos-base without having to cd into the base/
subdirectory.

Usage:
    python install_base.py install -- [FLAGS]

This is equivalent to:
    cd base/ python setup.py install -- [FLAGS]
"""

import os
import sys
import subprocess


def main():
    # Setup pwd to `base` dir
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(root_dir, "base")

    if not os.path.exists(base_dir):
        print(f"Error: base directory not found at {base_dir}")
        sys.exit(1)

    setup_py = os.path.join(base_dir, "setup.py")
    if not os.path.exists(setup_py):
        print(f"Error: setup.py not found at {setup_py}")
        sys.exit(1)
    original_dir = os.getcwd()
    os.chdir(base_dir)

    try:
        # Run setup.py with all the arguments passed to this script
        cmd = [sys.executable, "setup.py"] + sys.argv[1:]
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
