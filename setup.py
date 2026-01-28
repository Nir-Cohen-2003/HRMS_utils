import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools_rust import Binding, RustExtension


class CustomBuildExt(build_ext):
    def run(self):
        subprocess.run(["cargo", "clean"], check=True)
        super().run()


setup(
    name="hrms_utils",
    version="0.8.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    rust_extensions=[
        RustExtension("hrms_utils._rust", path="Cargo.toml", binding=Binding.PyO3)
    ],
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
