import os
import io
import subprocess
import time
from contextlib import redirect_stdout


def main():
    cwd = os.getcwd()
    subprocess.run(["rm", "-rf", "pk_cpp"], cwd=cwd)

    try:
        del os.environ["PK_LOOP_FUSE"]
    except Exception as e:
        print("TRIED DELETING ENV VARIABLE, BUT FAILED:\n", e)

    result_vanilla = subprocess.run(["python", "test_loop_fusion.py"], cwd=cwd, capture_output=True, text=True)
    
    # get output
    vanilla_out = result_vanilla.stdout

    # Again but with env variable
    os.environ["PK_LOOP_FUSE"] = "1"
    # remove old compilations
    subprocess.run(["rm", "-rf", "pk_cpp"], cwd=cwd)

    print("rerunning... ")
    result_fused = subprocess.run(["python", "test_loop_fusion.py"], cwd=cwd, capture_output=True, text=True)

    fused_out = result_fused.stdout

    if vanilla_out != fused_out:
        print("[X] MISMATCHED OUTPUTS:")
        print("\t[-] WITHOUT FUSION:")
        print(vanilla_out)
        print("\n\t[+] WITH FUSION:")
        print(fused_out)

if __name__ == "__main__":
    main()