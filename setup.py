from setuptools import find_packages, setup

with open("docs/version.txt", "r") as f:
    version = f.read().strip()

setup(
    name="vision",
    version=version,
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
    ],
)
