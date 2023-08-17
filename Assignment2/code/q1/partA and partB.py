# Algorithm:
#   Method of rejection:
#       Take that is valid from a data that already satisfies our prvious conditions.
#   Description:
#       Here we take a uniformly distributed points in a rectangle which is fairly simple
#   and more over easy to generate. Thus we have the data which satisfies one of our 
#   conditions which is the uniform spread. Now we just discard all the points which 
#   are outside of our desired ellipse range which leaves a data set of uniformly spread
#   points within an ellipse.

#   Downfall of algorithm:
#       If we are generating, say 100, points theoritically it may run forever and that is
#   precisily because we are looping until we reach 100 succesful points. Nevertheless 
#   practically it won't be the case because the random funtion numpy.random.uniform() is still
#   a pseudo random number generator and there isn't any nartural randomness to only favour
#   the luck to only get points outside the ellipse in the initial dataset generating the 
#   uniform rectangle spread points.

import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
np.random.seed(19680801)

# Total samples required
N=10000000

# rejection sampling

# major (a) and minor (b) axis lengths
a=2
b=1

x=[]
y=[]
i=0
while i<N :
    temp_x = np.random.uniform(low=-1*a,high=a)
    temp_y = np.random.uniform(low=-1*b,high=b)
    inORout = temp_x**2/a**2 + temp_y**2/b**2 # Checking whether the generated point is within the ellipse or not
    if inORout <= 1 :
        x.append(temp_x)
        y.append(temp_y)
        i += 1

# plot setup
xmin = np.min(x)
xmax = np.max(x)

ymin = np.min(y)
ymax = np.max(y)

xbins = np.linspace(xmin-5,xmax+5,1000)
ybins = np.linspace(ymin-5,ymax+5,1000)

plt.hist2d(x,y,bins=[xbins,ybins])
plt.show()