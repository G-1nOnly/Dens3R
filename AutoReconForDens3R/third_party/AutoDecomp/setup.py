from pathlib import Path

from setuptools import find_packages, setup

description = ["Automatic object localization from casual object-centric captures"]

root = Path(__file__).parent
with open(str(root / "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()
with open(str(root / "auto_decomp/__init__.py"), "r") as f:
    version = eval(f.read().split("__version__ = ")[1].split()[0])
with open(str(root / "requirements.txt"), "r") as f:
    dependencies = f.read().split("\n")

setup(
    name="auto_decomp",
    version=version,
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=dependencies,
    author="Yuang Wang",
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zju3dv/AutoRecon",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
