# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name='Brain_Corp',
    version='1.0.0',
    packages=find_packages(include=['phone_detection.*'])
)