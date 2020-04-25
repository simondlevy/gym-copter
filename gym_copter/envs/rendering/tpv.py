#!/usr/bin/env python3
'''
Adapted from

https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
'''

import threading
import time
import matplotlib.pyplot as plt
import numpy as np

class TPV:

    def __init__(self):

        self.state = None

        thread = threading.Thread(target=self._update)
        thread.daemon = True
        thread.start()

        x = np.linspace(-np.pi, +np.pi, 1000)
        plt.plot(x, np.sin(x))
        plt.show()

    def display(self, mode, state):

        self.state = state
       
    def _update(self):

        while True:

            continue
