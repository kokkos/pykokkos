import argparse
import sys
from typing import Tuple


def check_sizes(N: int, M: int, S: int, E: int, nrepeat: int) -> Tuple[int, int, int, int, int]:
    # If S is undefined and N or M is undefined, set S to 2^18 or the bigger of N and M.
    if S == -1 and (N == -1 or M == -1):
        S = 2 ** 18
        if S < N:
            S = N
        if S < M:
            S = M

    # If S is undefined and both N and M are defined, set S = N * M.
    if S == -1:
        S = N * M

    # If both N and M are undefined, fix row length to the smaller of S and 2 ^ 10 = 1024.
    if N == -1 and M == -1:
        if S > 1024:
            M = 1024
        else:
            M = S

    # If only M is undefined, set it.
    if M == -1:
        M = int(S / N)

    # If N is undefined, set it.
    if N == -1:
        N = int(S / M)

    # If E is undefined, set it to 2 ^ 10 = 1024.
    if E == -1:
        E = 2 ** 10

    print(f"  Total size S = {S} N = {N} M = {M} E = {E}")

    # Check sizes.
    if S < 0 or N < 0 or M < 0 or nrepeat < 0:
        print("  Sizes must be greater than 0.")
        sys.exit(1)

    if N * M != S:
        print("  N * M != S\n")
        sys.exit(1)

    return N, M, S, E, nrepeat


def parse_args() -> Tuple[int, int, int, int, int, str, bool]:
    N: int = -1
    M: int = -1
    S: int = -1
    E: int = -1
    nrepeat: int = 100
    space: str = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--rows", type=int)
    parser.add_argument("-M", "--columns", type=int)
    parser.add_argument("-S", "--size", type=int)
    parser.add_argument("-E", "--elements", type=int)
    parser.add_argument("-nrepeat", "--nrepeat", type=int)
    parser.add_argument("-space", "--execution_space", type=str)
    parser.add_argument("--fill", action="store_true",
                        help="Specify whether to use ViewType.fill() or"
                        " to initialize with sequential for loop")
    args = parser.parse_args()

    if args.rows:
        N = 2 ** args.rows
    if args.columns:
        M = 2 ** args.columns
    if args.size:
        S = 2 ** args.size
    if args.elements:
        E = 2 ** args.elements
    if args.nrepeat:
        nrepeat = args.nrepeat
    if args.execution_space:
        space = args.execution_space

    return check_sizes(N, M, S, E, nrepeat) + (space, args.fill)
