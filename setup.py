import sys
from setuptools import setup

if sys.version_info < (3, 5):
    raise Exception('python 3.5 or newer is required')

# read version.py
__version__ = None
with open('psfrec/version.py') as f:
    exec(f.read())

setup()
