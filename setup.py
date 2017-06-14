"""
Installation script for pySTATIS
"""

from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()
setup(
    name='pySTATIS',

    version='0.2.0',
    description='Python implementation of STATIS for analysis of several example_data tables',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/mfalkiewicz/pySTATIS',

    # Author details
    author='Marcel Falkiewicz',

    # Choose your license
    license='Apache',
    packages=['pySTATIS'],
    py_modules=["statis"],
    
    install_requires=['numpy', 'scipy']
)

