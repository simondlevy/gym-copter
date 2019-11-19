#!/usr/bin/env python

# https://www.mail-archive.com/pyglet-users@googlegroups.com/msg02182.html

import pyglet
from pyglet.gl import *

window = pyglet.window.Window()

@window.event
def on_draw():

    window.clear()

    glLoadIdentity()
    label1.draw()

    # The rotation occurs around the origin (lower left corner)
    # so first we 'move' the origin to the center of the window
    # and since the label is anchored in the center of the label,
    # the text will seem to rotate around its center.
    glTranslatef(window.width // 2, window.height // 2, 0.0)
    glRotatef(90.0, 0.0, 0.0, 1.0)
    label2.draw()

    glLoadIdentity()
    glTranslatef(100.0, window.height - 100, 0.0)
    glRotatef(45.0, 0.0, 0.0, 1.0)
    label3.draw()


label1 = pyglet.text.Label('Not rotated',x=10, y=10)

label2 = pyglet.text.Label('Rotated 90 degrees',anchor_x='center', anchor_y='center')

label3 = pyglet.text.Label('Rotated 45 degrees',anchor_x='center', anchor_y='center')

pyglet.app.run()
