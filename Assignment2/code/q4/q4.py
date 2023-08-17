from h5py._hl import datatype
import numpy as np
import matplotlib.pyplot as plt
import h5py
import statistics
import math
from numpy.core.fromnumeric import shape



# resizing and handling data
input_data = h5py.File('../data/mnist.mat','r')
labels=np.array(input_data['labels_train'])
digits=np.array(input_data['digits_train'])
resize_input=[]
mean_data=[]
cov_matrix= []
cov_matrix_ = []
for i in range(0,60000):
    resize_input.append(digits[i].reshape(784,order='F'))
mean_digit=np.zeros((10,784),dtype=float)
digit_count=[0,0,0,0,0,0,0,0,0,0]
data=[]

# finding mean for every digit and keeping it as list 
for j in range(0,10):
    row=[]
    for i in range(0,len(labels[0])):
        if(labels[0][i]==j):
            mean_digit[j]=mean_digit[j]+resize_input[i]
            row.append(resize_input[i])
            digit_count[j]=digit_count[j]+1
    mean_digit[j]=mean_digit[j]/float(digit_count[j])
    data.append(row)
    mean_data.append(data[j]-mean_digit[j])
    cov_matrix_.append(np.cov(mean_data[j],rowvar = False))


#part -2 caluculating covariance matrix 
# by multiplicating the deviation vector transpose with itself 


for digit in range(0,10):
    new_data = np.array(data[digit])
    new_data = new_data.transpose()
    dev_vec = np.empty(0)
    for i in range(0,784):
        value = statistics.stdev(new_data[i],mean_digit[digit][i])
        dev_vec = np.append(dev_vec,value)
    dev_vec = np.reshape(dev_vec,(784,1))
    cov = np.dot(dev_vec,dev_vec.transpose())
    cov_matrix.append(cov)

eigen_values = []      # list of eigen values for each digit 
eigen_vectors = []      # list of eigen vectors for each digit 
max_eigen_value = []    # list of max eigen values of each digit 
max_eigen_vec = []      # list of the max eigen vectors for each digit 
modes = np.empty(0)


# part 3 and part 4 caluculating principal modes and plotting eigen values 


for i in range(0,10):
    eigen_values_,eigen_vectors_ = np.linalg.eigh(cov_matrix_[i])
    sorted_index = np.argsort(eigen_values_)[::-1]
    eigen_values_ = eigen_values_[sorted_index]
    eigen_vectors_ = eigen_vectors_[:,sorted_index]
    eigen_values.append(eigen_values_)
    eigen_vectors.append(eigen_vectors_)
    max_eigen_value.append(eigen_values_[0])
    max_eigen_vec.append(eigen_vectors_[0])
    m=0
    for k in range(0,784):
        if(max_eigen_value[i]/100 < eigen_values_[k]):
            m = m+1
        else:
            break
    modes = np.append(modes,m)
    x = np.arange(1,785)
    plt.plot(x,eigen_values_)
    plt.xlabel("x-axis")
    plt.ylabel("eigen value")
    plt.title(f"Eigen Values for {i}")
    plt.show()
modes = np.reshape(modes,(10,1))
print(modes) # getting output as {56.,27.,86.,83.,69.,60.,63.,89.,63.}


#part - 5



for digit in range(0,10):                                         # displaying all mean_data for each digit 
        temp_img = mean_digit[digit].reshape(28,28)
        plt.imshow(temp_img,cmap="gray")
        plt.title(f"q4_{digit}_mean_img.jpg")
        plt.show()

        temp = np.subtract( mean_digit[digit],np.multiply(max_eigen_vec[digit],math.sqrt(max_eigen_value[digit])))   #displaying image with removing the principal mode
        temp = temp.reshape(28,28)
        plt.imshow(temp,cmap="gray")
        plt.title(f"q4_{digit}_reduced_mean_img.jpg")
        plt.show()

        
        temp = np.subtract( mean_digit[digit],np.multiply(max_eigen_vec[digit],-1*math.sqrt(max_eigen_value[digit])))  #displaying image with adding more value to principal mode 
        temp = temp.reshape(28,28)
        plt.imshow(temp,cmap="gray")
        plt.title(f"q4_{digit}_added_mean_img.jpg")
        plt.show()