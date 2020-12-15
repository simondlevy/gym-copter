#!/usr/bin/env python3

'''
Python distutils setup file for gym-copter module.

Copyright (C) 2019 Simon D. Levy

MIT License
'''

#from distutils.core import setup
from setuptools import setup

setup (name = 'gym_copter',
    version = '0.1',
    install_requires = ['gym', 'numpy', 'box2d-py'],
    description = 'Gym environment for multicopters',
    packages = ['gym_copter', 'gym_copter.envs', 'gym_copter.dynamics', 'gym_copter.rendering'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/gym-copter',
    license='MIT',
    platforms='Linux; Windows; OS X'
    )
