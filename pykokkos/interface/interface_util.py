import sys


def get_lineno(frame):
    """
    Return current line number `inspect` frame
    """
    return frame.f_lineno


def get_filename(frame):
    """
    Return current line number `inspect` frame
    """
    return frame.f_code.co_filename


def generic_error(filename: str, lineno: str | int, error: str, exit_message: str):
    print(f"\n\033[31m\033[01mError {filename}:{lineno}\033[0m: {error}")
    sys.exit(f"PyKokkos: {exit_message}")
