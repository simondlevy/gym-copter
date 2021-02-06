#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def main():


    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect(1)

    def animate(i):

        return [ax.add_patch(plt.Circle((0.5, 0.5), 0.45, color='r'))]

    anim = animation.FuncAnimation(fig, animate, frames=10, interval=20, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
