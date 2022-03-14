from setuptools import setup, find_packages
from xarray_extensions import VERSION

setup(
    name="xarray_extensions",
    version=VERSION,
    packages=find_packages(
        exclude=[
            "test",
            "test.*",
            "docs",
            "docs.*"
        ]
    ),
    package_data={
    },
    zip_safe=True
)