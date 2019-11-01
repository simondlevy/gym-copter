#!/usr/bin/env python3

'''
Python distutils setup file for gym-copter module.

Copyright (C) 2019 Simon D. Levy

MIT License
'''

#from distutils.core import setup
from setuptools import setup

setup (name = 'gym-copter',
    version = '0.1',
    install_requires = ['tensorflow', 'gym'],
    description = 'Gym environment for multicopters',
    packages = ['gym-copter'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/gym-copter',
    license='MIT',
    platforms='Linux; Windows; OS X'
    )
