from setuptools import setup, find_packages

VERSION = '1.0'
DESCRIPTION = 'A python package implementing efficient multimodal learning'

setup(
    name="eff-mml",
    version=VERSION,
    author="Siddharth Joshi, Arnav Jain",
    author_email="sjoshi804@cs.ucla.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=DESCRIPTION,
    packages=find_packages()
)