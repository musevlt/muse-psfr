import sys

from setuptools import setup

if sys.version_info < (3, 6):
    raise Exception('python 3.6 or newer is required')

setup(use_scm_version=True)
