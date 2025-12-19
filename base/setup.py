#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import platform
from skbuild import setup

# some Cray systems default to static libraries and the build
# will fail because BUILD_SHARED_LIBS will get set to off
if os.environ.get("CRAYPE_VERSION") is not None:
    os.environ["CRAYPE_LINK_TYPE"] = "dynamic"

cmake_args = [
    f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
    f"-DPython3_EXECUTABLE:FILEPATH={sys.executable}",
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON",
]

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action="store_true")


def set_cmake_bool_option(opt, enable_opt, disable_opt):
    try:
        if enable_opt:
            cmake_args.append(f"-D{opt}:BOOL=ON")
        if disable_opt:
            cmake_args.append(f"-D{opt}:BOOL=OFF")
    except Exception as e:
        print(f"Exception: {e}")


def add_arg_bool_option(lc_name, disp_name, default=None):
    # enable option
    parser.add_argument(
        f"--enable-{lc_name}",
        action="store_true",
        default=default,
        help=f"Explicitly enable {disp_name} build",
    )
    # disable option
    parser.add_argument(
        f"--disable-{lc_name}",
        action="store_true",
        help=f"Explicitly disable {disp_name} build",
    )


# add options
add_arg_bool_option("experimental", "ENABLE_EXPERIMENTAL")
add_arg_bool_option("layouts", "ENABLE_LAYOUTS")
add_arg_bool_option("memory-traits", "ENABLE_MEMORY_TRAITS")
add_arg_bool_option("thin-lto", "ENABLE_THIN_LTO")
add_arg_bool_option("werror", "ENABLE_WERROR")
add_arg_bool_option("timing", "ENABLE_TIMING")
parser.add_argument(
    "--cxx-standard",
    default=17,
    type=int,
    choices=[14, 17, 20],
    help="Set C++ language standard",
)
parser.add_argument(
    "--enable-view-ranks",
    default=None,
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 7],
    help="Maximum number of concrete view ranks",
)
parser.add_argument(
    "--kokkos-root",
    default=None,
    type=str,
    help="Path to kokkos install prefix",
)
parser.add_argument(
    "--cmake-args",
    default=[],
    type=str,
    nargs="*",
    help="{}{}{}".format(
        "Pass arguments to cmake. Use w/ pip installations",
        "and --install-option, e.g. --install-option=--cmake-args=",
        '"-DENABLE_LAYOUTS=ON -DKokkos_DIR=/usr/local/lib/cmake/Kokkos"',
    ),
)

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1] + left

set_cmake_bool_option(
    "ENABLE_EXPERIMENTAL", args.enable_experimental, args.disable_experimental
)
set_cmake_bool_option("ENABLE_LAYOUTS", args.enable_layouts, args.disable_layouts)
set_cmake_bool_option(
    "ENABLE_MEMORY_TRAITS",
    args.enable_memory_traits,
    args.disable_memory_traits,
)
set_cmake_bool_option("ENABLE_THIN_LTO", args.enable_thin_lto, args.disable_thin_lto)
set_cmake_bool_option("ENABLE_WERROR", args.enable_werror, args.disable_werror)
set_cmake_bool_option("ENABLE_TIMING", args.enable_timing, args.disable_timing)

cmake_args.append(f"-DCMAKE_CXX_STANDARD={args.cxx_standard}")

for itr in args.cmake_args:
    cmake_args += itr.split()

if args.enable_view_ranks is not None:
    cmake_args += [f"-DENABLE_VIEW_RANKS={args.enable_view_ranks}"]

if args.kokkos_root is not None:
    os.environ["CMAKE_PREFIX_PATH"] = ":".join(
        [args.kokkos_root, os.environ.get("CMAKE_PREFIX_PATH", "")]
    )

if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    darwin_version = platform.mac_ver()[0].split(".")
    darwin_version = ".".join([darwin_version[0], darwin_version[1]])
    cmake_args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={darwin_version}"]

# DO THIS LAST!
# support PYKOKKOS_BASE_SETUP_ARGS environment variables because
#  --install-option for pip is a pain to use
# PYKOKKOS_BASE_SETUP_ARGS should be space-delimited set of cmake arguments, e.g.:
#   export PYKOKKOS_BASE_SETUP_ARGS="-DENABLE_LAYOUTS=OFF -DENABLE_MEMORY_TRAITS=ON"
env_cmake_args = os.environ.get("PYKOKKOS_BASE_SETUP_ARGS", None)
if env_cmake_args is not None:
    cmake_args += env_cmake_args.split(" ")


# --------------------------------------------------------------------------- #
#
def get_project_version():
    # open "VERSION"
    with open(os.path.join(os.getcwd(), "VERSION"), "r") as f:
        data = f.read().replace("\n", "")
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


# --------------------------------------------------------------------------- #
#
def get_long_description():
    long_descript = ""
    try:
        long_descript = open("README.md").read()
    except Exception:
        long_descript = ""
    return long_descript


# --------------------------------------------------------------------------- #
#
def parse_requirements(fname="requirements.txt"):
    _req = []
    requirements = []
    # read in the initial set of requirements
    with open(fname, "r") as fp:
        _req = list(filter(bool, (line.strip() for line in fp)))
    # look for entries which read other files
    for itr in _req:
        if itr.startswith("-r "):
            # read another file
            for fitr in itr.split(" "):
                if os.path.exists(fitr):
                    requirements.extend(parse_requirements(fitr))
        else:
            # append package
            requirements.append(itr)
    # return the requirements
    return requirements


# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
#       is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    print("CMake arguments: {}".format(" ".join(cmake_args)))
    setup(
        name="pykokkos-base",
        packages=["kokkos"],
        version=get_project_version(),
        cmake_args=cmake_args,
        cmake_languages=("C", "CXX"),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        install_requires=parse_requirements("requirements.txt"),
        project_urls={"kokkos": "https://github.com/kokkos/kokkos"},
    )
