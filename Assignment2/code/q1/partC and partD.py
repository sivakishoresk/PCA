# Algorithm:
#       Here we generate a point in a rectangle and according to it's one coordinate
#   we shift the other coordinate so they all endup in our desired parallellogram and
#   therefore in our desired triangle.
#       It more like generating uniform points in a rectangle and modelling it's shape
#   to match our parallelogram. And then divide the parallellogram in half to get uniform
#   distribution in a desired triangle.

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import math

# Setting seed for reproducibility
np.random.seed(19680801)

# Total samples required
N=10000000

# Shift sampling

# Given points
a = [0,0]
b = [np.pi,0]
c = [np.pi/3,np.exp(1)]

x=[]
y=[]

# First get our parallelogram
for i in range(N):
    temp_x = np.random.uniform(low=a[0],high=b[0])
    temp_y = np.random.uniform(low=a[0],high=c[1])
    temp_x += temp_y*c[0]/c[1]
    if temp_x > b[0] - (temp_y*(b[0]-c[0])/c[1]):
        continue
    x.append(temp_x)
    y.append(temp_y)


# plot setup
xmin = np.min(x)
xmax = np.max(x)

ymin = np.min(y)
ymax = np.max(y)

xbins = np.linspace(xmin-5,xmax+5,1000)
ybins = np.linspace(ymin-5,ymax+5,1000)

plt.hist2d(x,y,bins=[xbins,ybins])
plt.show()