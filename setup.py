import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    raise Exception('python 3.5 or newer is required')

# read version.py
__version__ = None
with open('psfrec/version.py') as f:
    exec(f.read())

setup(
    name='psfrec',
    version=__version__,
    description='MUSE WFM-AO PSF reconstruction from SPARTA',
    author='Simon Conseil',
    author_email='simon.conseil@univ-lyon1.fr',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=['mpdaf', 'astropy', 'scipy', 'numpy', 'joblib'],
    extras_require={
        'all': ['matplotlib', 'colorama'],
    },
    entry_points={
        'console_scripts': [
            'psfrec=psfrec.run_psfrec:main',
        ]
    },
)
