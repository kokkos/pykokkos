#!/usr/bin/env python
"""
Wrapper script to install pykokkos-base from the root pykokkos directory.
This script allows installing pykokkos-base without having to cd into the base/
subdirectory.

Usage:
    python install_base.py install -- [FLAGS]

This is equivalent to:
    cd base/ python -m pip install .  (with PYKOKKOS_BASE_SETUP_ARGS set)
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
    original_dir = os.getcwd()
    os.chdir(base_dir)

    try:
        argv = sys.argv[1:]
        if not argv:
            raise SystemExit(
                "Usage: python install_base.py install -- -D<...> [other CMake -D flags]"
            )

        # Allow users to pass verbosity flags through this wrapper.
        pip_verbose = ("--verbose" in argv) or ("-v" in argv)

        # Parse CMake flags after `--`.
        #
        # Typical invocation:
        #   python install_base.py install -- -DENABLE_LAYOUTS=ON -DENABLE_VIEW_RANKS=3
        cmake_flags = []
        if "--" in argv:
            idx = argv.index("--")
            cmake_flags = argv[idx + 1 :]
        else:
            # Backwards-compat: if someone calls `python install_base.py install -D...`
            # treat everything except the leading `install` as CMake flags.
            cmake_flags = [a for a in argv if a != "install"]

        # Ensure wrapper verbosity flags don't leak into CMake arguments.
        # These are only meant to affect `pip install`, not the -D... CMake args.
        cmake_flags = [a for a in cmake_flags if a not in ("-v", "--verbose")]

        env = os.environ.copy()
        existing = env.get("PYKOKKOS_BASE_SETUP_ARGS", "").strip()
        if cmake_flags:
            cmake_str = " ".join(cmake_flags).strip()
            env["PYKOKKOS_BASE_SETUP_ARGS"] = (
                f"{existing} {cmake_str}".strip() if existing else cmake_str
            )

        cmd = [sys.executable, "-m", "pip", "install", "."]
        if pip_verbose:
            cmd.append("--verbose")

        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
