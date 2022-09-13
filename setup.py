from setuptools import setup, find_packages

setup(
    name="pykokkos",
    version="0.1",
    packages=find_packages(include=["pykokkos", "pykokkos.*"]),
    include_package_data=True,
    package_data={"": ["*.sh"]}
)
