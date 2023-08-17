# Part A is in the report which asked for an implementable algorithm. Algorithm is used in this code and
# explaination is given below and also written in the report.

# Principle Component Analaysis:
#       This mainly focuses on reducing the dimentionality at the least expense of accuracy/loss of information
#       Here from the scatter plot we can say that our original components X and Y are positive and lineaarly
#   correlated and we can find a principle component from the eigen vectors of it's covariance which can give
#   us the best approximation of linear relationship between the two original components because it captures
#   most of the distributions spread and the spread perpendicular to this eigen vector direction is minimal
#   which can be seen directly by the scatter plot or by analysing the corresponding eigen directions eigen value.
#       We can implement an algorithm two ways, one is which involves the calculation covariance matrix and their 
#   eigen vectors, other is by finding the line where the sum of squares of perpendicular distances from points to 
#   that line which passes through mean will give the best approximate realtion between X and Y. Althought both results
#   the same linear relation in X and Y.


def covariance(arr1, arr2, n):
    sum = 0.
    for i in range(0, n):
        sum = (sum + (arr1[i] - mean(arr1, n)) * (arr2[i] - mean(arr2, n)))
    return sum/(n-1)

def mean(arr, n):
    sum = 0.
    for i in range(0, n):
        sum += arr[i]
    return sum/n



import matplotlib.pyplot as plt
import numpy as np
import h5py as hpy

#reading data set 1
d = hpy.File('points2D_Set1.mat','r')

# Processing data for a good format
data_x = np.array(d.get('x'))
data_y = np.array(d.get('y'))
data1 = np.array([data_x[0],data_y[0]])

#calculating covariance matrix, mean vector, eigen values and eigen vectors
n = np.shape(data1)[1]
covar=[[covariance(data1[0],data1[0],n),covariance(data1[0],data1[1],n)],[covariance(data1[0],data1[1],n),covariance(data1[1],data1[1],n)]]
eig_val, eig_vec = np.linalg.eigh(covar)
mean_data1 = [mean(data1[0],n),mean(data1[1],n)]

#plot setup
slope = eig_vec[1][1]/eig_vec[1][0]
x = np.linspace(np.min(data1[0]), np.max(data1[0]),1000)

# The required best approximate linear relationship between x and y
y = slope*x - slope*mean_data1[0] + mean_data1[1]
plt.plot(x,y,color='red')

plt.scatter(data1[0],data1[1])
plt.show()


plt.figure()

#reading data set 2
d = hpy.File('points2D_Set2.mat','r')

# Processing data for a good format
data_x = np.array(d.get('x'))
data_y = np.array(d.get('y'))
data2 = np.array([data_x[0],data_y[0]])

#calculating covariance matrix, mean vector, eigen values and eigen vectors
n = np.shape(data2)[1]
covar=[[covariance(data2[0],data2[0],n),covariance(data2[0],data2[1],n)],[covariance(data2[0],data2[1],n),covariance(data2[1],data2[1],n)]]
eig_val, eig_vec = np.linalg.eigh(covar)
mean_data2 = [mean(data2[0],n),mean(data2[1],n)]

#plot setup
slope = eig_vec[1][1]/eig_vec[1][0]
x = np.linspace(np.min(data2[0]), np.max(data2[0]),1000)

# The required best approximate linear relationship between x and y
y = slope*x - slope*mean_data2[0] + mean_data2[1]
plt.plot(x,y,color='red')

plt.scatter(data2[0],data2[1])
plt.show()