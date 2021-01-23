'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import os

from setuptools import find_packages, setup

from activations import __version__

install_requires = "torch"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="activations",
    version=__version__,
    author="densechen",
    author_email="densechen@foxmail.com",
    description="activations: a package contains different kinds of activation functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/densechen/AReLU",
    download_url = 'https://github.com/densechen/AReLU/archive/master.zip',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.6',
)
