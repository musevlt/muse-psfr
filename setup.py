import sys
from setuptools import setup

if sys.version_info < (3, 5):
    raise Exception('python 3.5 or newer is required')

setup(use_scm_version=True)
