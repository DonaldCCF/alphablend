#! /usr/bin/env python

from setuptools import setup, find_packages
import factorgene

DESCRIPTION = __doc__
VERSION = factorgene.__version__

setup(name='factorgene',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open("README.md").read(),
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10'],
      author='Donald Cheng',
      author_email='cu30ccf@gmail.com',
      url='https://github.com/donaldccf/FactorGene',
      license='new BSD',
      packages=find_packages(exclude=['*.tests',
                                      '*.tests.*']),
      zip_safe=False,
      package_data={'': ['LICENSE']},
      install_requires=['scikit-learn>=1.0.2',
                        'joblib>=1.0.0'])
