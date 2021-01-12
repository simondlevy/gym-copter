#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

neat:
	neat-evolve.py --env gym_copter:Lander3D-v0 --ngen 100

hyper:
	neat-evolve.py --env gym_copter:Lander3D-v0 --ngen 100 --hyper

es:
	neat-evolve.py --env gym_copter:Lander3D-v0 --ngen 100 --eshyper

clean:
	rm -rf models/ visuals/
