"""
    Feature Selection for Clustering
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fsfc',
    version='0.0.1',
    description='Feature Selection for Clustering',
    long_description=long_description,
    url='https://github.com/danilkolikov/fsfc',
    author='Danil Kolikov',
    author_email='danilkolikov@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),

    install_requires=['scikit-learn', 'numpy'],

    extras_require={
        'test': [],
    },
)
