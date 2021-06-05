#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

2d:
	neat-evolve.py config/lander2d

3d:
	neat-evolve.py config/lander3d

best:
	ls -c1 models | head

clean:
	rm -rf models/ visuals/ runs/ *.mp4 *.csv
