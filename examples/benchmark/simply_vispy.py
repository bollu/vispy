#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import time
from vispy import app, use
from vispy.gloo import clear

use('pyqt4')
# use('glut')
# use('pyglet')

canvas = app.Canvas(size=(512, 512), title = "Do nothing benchmark (vispy)",
                    keys='interactive')


@canvas.connect
def on_draw(event):
    global t, t0, frames
    clear(color=True, depth=True)

    t = time.time()
    frames = frames + 1
    elapsed = (t - t0)  # seconds
    if elapsed > 2.5:
        print("FPS : %.2f (%d frames in %.2f second)"
              % (frames / elapsed, frames, elapsed))
        t0, frames = t, 0
    canvas.update()

t0, frames, t = time.time(), 0, 0
canvas.show()
app.run()
