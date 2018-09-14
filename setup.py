import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    raise Exception('python 3.5 or newer is required')

setup(
    name='psfrec',
    version='0.1',
    description='MUSE WFM-AO PSF reconstruction from SPARTA',
    author='Simon Conseil',
    author_email='simon.conseil@univ-lyon1.fr',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['mpdaf', 'astropy', 'scipy', 'matplotlib', 'numpy'],
    entry_points={
        'console_scripts': [
            'psfrec=psfrec.run_psfrec:main',
        ]
    },
)
