import setuptools

NAME = "kazane"
VERSION = '1.0'
AUTHOR = 'Chin-Yun Yu'
EMAIL = 'lolimaster.cs03@nctu.edu.tw'


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/kazane",
    packages=setuptools.find_packages(),
    install_requires=['torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
