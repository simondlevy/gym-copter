#!/usr/bin/env python3

'''
Python distutils setup file for gym-copter module.

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='gym_copter',
    version='0.1',
    install_requires=['gymnasium', 'numpy', 'matplotlib'],
    description='Gym environment for multicopters',
    packages=['gym_copter',
              'gym_copter.envs',
              'gym_copter.cmdline',
              'gym_copter.sensors',
              'gym_copter.sensors.vision'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/gym-copter',
    license='MIT',
    platforms='Linux; Windows; OS X'
    )
