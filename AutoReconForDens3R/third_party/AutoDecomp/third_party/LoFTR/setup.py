from pathlib import Path

from setuptools import find_packages, setup

description = ["LoFTR: Detector-Free Local Feature Matching with Transformers"]

root = Path(__file__).parent
with open(str(root / "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="LoFTR",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
