# Algorithm:
#   General form of a multivariate matrix M would be
#                   M = A*W + mean
#       where W is a matrix formed by i.i.d independent standard gaussian ditributions.
#
#   Let M' be M-mean.
#       M' = A*W
#
#   Let M'=[m1,m2] and W=[w1,w1]
#       From linear algebra we know that adding/subtracting one column from the other doesn't change
#   it's eigen values nor eigen vectors so essentially we can obtain a lower triangular matrix for A
#   and it works the same as the real/original A.
#   
#       If we are given a co-variance of a multivariate gaussian which is essentially the same as giving
#   us the A which we require in order to convert two standard un-correlated gaussian distributions to their
#   corresponding correlated multivariate distribution.
#       This is because for a multi-variate gaussian of type "A*W + mean" covariance matrix is A*A`. From this
#   we can obtain all the values assuming A to be a lower triangle and multiplying with it's transpose gives
#   us n equations in n variables and we can find all these variables algebraically and obtain A.
#       This is what is precisely done by the cholesky of linalg module of numpy. Or for our pupose we can define
#   our own for a bi-variate as it's the most visuvalizable data in sense of plotting and representation.
#
# Although can be generalised but sticking to for question. Generalised process is just calculation.

import numpy as np
import matplotlib.pyplot as plt
import math

def chlsky(matrix):
    a = math.sqrt(matrix[0][0])
    b = matrix[0][1]/a
    c = math.sqrt(matrix[1][1] - b**2)
    return np.array([[a,0.],[b,c]])

def norm(matrix):
    sum = 0.
    for i in matrix:
        for j in i:
            sum += j**2
    return math.sqrt(sum)




# Sample size
N=1000

# Setting seed for reproducibility
np.random.seed(19680801)

# Given mean and co-variance matrices
mean = [1,2]
covar = [[1.6250, -1.9486],[-1.9486,3.8750]]
covar_norm = norm(covar)

error = []
while N < 1000000:
    error_per_sample = []
    for i in range(100):
        x = np.random.normal(0,1,size=N)
        y = np.random.normal(0,1,size=N)

        #cky = np.linalg.cholesky(covar)     # numpy's cholesky
        cky2 = chlsky(covar)                # my cholesky :)

        #corr_data = np.dot(cky, [x, y])    # numpy's cholesky
        corr_data = np.dot(cky2, [x, y])    # my cholesky :)

        mean_ML_X = np.sum(corr_data[0])/N
        mean_ML_Y = np.sum(corr_data[1])/N

        sum=np.array([[0.,0.],[0.,0.]])
        for elem in corr_data.transpose():
            sum[0][0] += (elem[0]-mean_ML_X)*(elem[0]-mean_ML_Y)
            sum[0][1] += (elem[0]-mean_ML_X)*(elem[1]-mean_ML_Y)
            sum[1][0] += (elem[1]-mean_ML_X)*(elem[0]-mean_ML_Y)
            sum[1][1] += (elem[1]-mean_ML_X)*(elem[1]-mean_ML_Y)
        covar_ML = sum/N
        e = norm(np.subtract(covar_ML,covar))/covar_norm
        error_per_sample += [e]
    error += [error_per_sample]
    N *= 10

xaxis = np.array(['1000','10000','100000'])
plt.boxplot(np.transpose(np.array(error)),labels=xaxis)
plt.show()