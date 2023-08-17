from PIL import Image
from numpy import asarray
import numpy as np
import statistics
import matplotlib.pyplot as plt 


total = np.empty((19200,0))
mean = np.zeros((19200,1))
for i in range(0,16):
    img = Image.open(f'../data/data_fruit/image_{i+1}.png')
    img = np.asarray(img)
    print(img.shape)
    img = img.reshape((19200,1))
    mean = mean + img
    total = np.append(total,img)
total = total.reshape(16,19200)
total = total.transpose()
mean = mean/16
mean = mean.astype(int)
print(mean)
print(mean.shape)

dev_vec = []
for i in range(0,19200):
    k = statistics.stdev(total[i])
    dev_vec.append(k)
dev_vec= np.array(dev_vec)
dev_vec = dev_vec.reshape(19200,1)
print(dev_vec,mean)
cov_matrix = np.dot(dev_vec,dev_vec.transpose()) # covaraince matrix caluculation


# caluculation of eigen values and eigen vectors 
eigen_values,eigen_vectors = np.linalg.eigh(cov_matrix)
sorted_index = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_index]
eigen_vectors = eigen_vectors[:,sorted_index]
eigen_vectors = eigen_vectors.astype(int)
eigen_vectors_4 = eigen_vectors[:,0:4]
eigen_values_10 = eigen_values[0:10]
print(eigen_values_10)
x = np.arange(1,11)
plt.plot(x,eigen_values_10)
plt.xlabel("X-axis")
plt.ylabel("Eigen Value")
plt.title("plot for first 10 eigen values")
plt.show()
plt.savefig("q6_eigen_value_plot.png")
plt.close()

mean = mean.reshape(80,80,3)
mean_img = plt.imshow(mean)
plt.show()





