import sys
import inspect

def get_lineno(frame):
    return frame.f_lineno

def get_filename(frame):
    return frame.f_code.co_filename

def generic_error(filename, lineno, error, exit_message):
    print(f"\n\033[31m\033[01mError {filename}:{lineno}\033[0m: {error}")
    sys.exit(f"PyKokkos: {exit_message}")