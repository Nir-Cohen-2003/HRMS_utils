import subprocess
from ctypes import pythonapi
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools_rust import Binding, RustExtension


class CustomBuildExt(build_ext):
    def run(self):
        # # remove all .so files within the src folder

        # # Try to locate the project's src directory and remove any .so files inside it.
        # src_dir = Path(__file__).parent.resolve() / "src"
        # if not src_dir.exists():
        #     src_dir = Path.cwd() / "src"

        # if src_dir.exists():
        #     for so_file in src_dir.rglob("*.so"):
        #         try:
        #             so_file.unlink()
        #             try:
        #                 # Inform the build system about the removed file
        #                 self.announce(f"Removed old shared library: {so_file}", level=2)
        #             except Exception:
        #                 # Fallback to printing if announce is unavailable
        #                 print(f"Removed old shared library: {so_file}")
        #         except OSError as exc:
        #             try:
        #                 self.announce(f"Could not remove {so_file}: {exc}", level=3)
        #             except Exception:
        #                 print(f"Could not remove {so_file}: {exc}")
        subprocess.run(["cargo", "clean"], check=True)  # cleaning old cargo artifacts
        super().run()


setup(
    name="hrms_utils",
    version="0.8.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    rust_extensions=[
        RustExtension(
            "hrms_utils._internal",
            path="Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        )
    ],
    python_requires=">=3.12",
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
