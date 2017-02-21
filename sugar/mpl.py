#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
## Filename:      MPL.py
## Version:       $Revision: 1.42 $
## Description:   ReStructured Text utilities
## Author:        Yannick Copin <yannick@ipnl.in2p3.fr>
## Created at:    2008-12-20 21:07:39
## Author:        $Author: ycopin $
## $Id: MPL.py,v 1.42 2014/04/30 18:54:40 ycopin Exp $
######################################################################

"""
.. _MPL:

ToolBox.MPL - Matplotlib utilities
==================================

.. Warning:: `MPL` imports `matplotlib`, but **should not** import
   `pylab` or `pyplot`.
"""

__author__ = "Yannick Copin <ycopin@ipnl.in2p3.fr>"
__version__ = "$Id: MPL.py,v 1.42 2014/04/30 18:54:40 ycopin Exp $"

import numpy as N
import matplotlib
from matplotlib.axes import Axes
from ToolBox.Misc import make_method
import pylab as P


class PointBrowser_TO_DELATE:
    """
    Click on a point to select and highlight it -- the object that
    generated the point will be shown in the upper left.  Use the
    'right' and 'left' arrows to browse through next and pervious
    points, 't' to tag current point and 'd' to delete all tags, 'v'
    to turn off browser visibility.

    Adapted from matplotlib data_browser example.

    >>> line, = ax.plot(x,y,'bo')
    >>> browser = PointBrowser(x,y,labels,line)
    """

    format = "Selected: %s"

    def __init__(self, x, y, labels, artist, picker=5):

        # Check length of input parameters
        assert len(x)==len(y)==len(labels), \
            'Error: incompatible x,y,labels lengthes (%i,%i,%i)' % \
            (len(x),len(y),len(labels))
        self.n = len(x)

        # Initalisation of given parameters
        self.x,self.y = N.array(x),N.array(y)
        self.labels = labels
        self.artist = artist

        # Axes and figure
        self.ax = self.artist.axes
        self.fig = self.ax.get_figure()

        # Picker radius
        self.artist.set_picker(picker)

        # Event connexions
        self.connect(pick_event=self.onpick, # Mouse interaction
                     key_press_event=self.onpress) # Keyboard interaction

        # Initialisation of the indice
        self.lastind = 0

        # Text and point to update
        self.text = self.ax.text(0.05, 0.95, self.format % (None),
                                 transform=self.ax.transAxes, va='top',
                                 visible=False)
        self.selected, = self.ax.plot([self.x[0]], [self.y[0]],
                                      'o', c='yellow', ms=15, mec='red',
                                      zorder=0, visible=False)
        # Tags
        self.tags = []

        self.LIST_TO_DELATE = []
        
    def connect(self, **kwargs):

        for key,val in kwargs.iteritems():
            self.fig.canvas.callbacks.connect(key,val)

    def onpress(self, event):
        """
        =====  ===========================
        Key    Action
        =====  ===========================
        right  next point
        left   previous point
        t      tag current point
        d      delete all tags
        v      turn-off browser visibility
        ?      help
        =====  ===========================
        """
        # If no event
        if self.lastind is None:
            return

        # Browse through next or previous point (periodic)
        if event.key=='right':  # Next
            self.lastind += 1
        elif event.key=='left': # Previous
            self.lastind -= 1
        elif event.key=='t':    # Add tag
            tag = self.ax.text(self.x[self.lastind], self.y[self.lastind],
                               ' '+self.labels[self.lastind],
                               rotation=45, ha='left', va='bottom')
            self.tags.append(tag)
        elif event.key=='d':    # Delete all tags
            for tag in self.tags:
                tag.remove()
        elif event.key=='v':    # Turn-off browser visibility
            self.selected.set_visible(False)
            self.text.set_visible(False)
            self.fig.canvas.draw()
            return
        elif event.key=='?':
            print self.onpress.__doc__
        else:
            return True
        self.lastind = self.lastind % self.n

        self.update()

    def onpick(self, event):
        """
        Update if a point is selected by mouse.
        """
        if event.artist != self.artist or not len(event.ind):
            return True

        # Click location
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # Find index of closest point
        distances = N.hypot(x-self.x[event.ind], y-self.y[event.ind])
        self.lastind = event.ind[distances.argmin()]

        self.update()

    def update(self):
        """Update the text and selected point"""

        if self.lastind is None:
            return

        self.selected.set_visible(True)
        self.selected.set_data(self.x[self.lastind], self.y[self.lastind])

        self.text.set_visible(True)
        self.text.set_text(self.format % self.labels[self.lastind])
        self.fig.canvas.draw()
        print self.labels[self.lastind]
        self.LIST_TO_DELATE.append(self.labels[self.lastind])        

if __name__=='__main__':

    x=N.array([-1,0,1])
    y=N.array([1,2,1])
    NAME=[0,1,2]
    scat=P.scatter(x,y,s=50)
    browser=PointBrowser_TO_DELATE(x, y,NAME,scat)
    P.show()
