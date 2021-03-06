#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/

2d:
	python3 gym_copter/envs/lander2d.py

3d:
	python3 gym_copter/envs/lander3d.py

viz:
	python3 gym_copter/envs/lander3d.py --visual

pose:
	python3 gym_copter/envs/lander3d.py --visual --freeze="0, 0, 5, 0, 0"
	# python3 gym_copter/envs/lander3d.py --visual --freeze="-3, -3, 5, 0, 0"

vs:
	python3 gym_copter/sensors/vision/vs.py

dvs:
	python3 gym_copter/sensors/vision/dvs.py

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 gym_copter/__init__.py
	flake8 gym_copter/dynamics/*.py
	flake8 gym_copter/rendering/*.py
	flake8 gym_copter/envs/*.py
	flake8 gym_copter/sensors/*.py
	flake8 gym_copter/sensors/vision/*.py
	flake8 drl/3dtest.py
	flake8 neat/3dtest.py
