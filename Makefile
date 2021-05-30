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
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/ *.csv

l2d:
	python3 gym_copter/envs/lander2d.py

l3d:
	python3 gym_copter/envs/lander3d.py

drop:
	python3 gym_copter/envs/drop3d.py

h2d:
	python3 gym_copter/envs/hover2d.py

h3d:
	python3 gym_copter/envs/hover3d.py

plot:
	python3 utils/copter-plot.py pid.csv

nodisp:
	python3 gym_copter/envs/lander3d.py --nodisplay

nopid:
	python3 gym_copter/envs/lander3d.py --nopid


vs:
	python3 gym_copter/envs/lander3d.py --vision

dvs:
	python3 gym_copter/envs/lander3d.py --dvs

pose:
	python3 gym_copter/envs/lander3d.py --visual --freeze="1, 0, 5, 30, 0"

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 utils/*.py
	flake8 gym_copter/__init__.py
	flake8 gym_copter/dynamics/*.py
	flake8 gym_copter/rendering/*.py
	flake8 gym_copter/envs/*.py
	flake8 gym_copter/sensors/*.py
	flake8 gym_copter/pidcontrollers/*.py
	flake8 drl/3dtest.py
	flake8 neat/3dtest.py
