#!/usr/bin/env python3
from examinimd import ExaMiniMD

def run() -> None:
    exa_mini_md = ExaMiniMD()
    exa_mini_md.init()
    exa_mini_md.run(exa_mini_md.input.nsteps)

if __name__ == "__main__":
    run()