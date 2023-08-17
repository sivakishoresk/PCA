import numpy as np
import matplotlib.pyplot as plt
import h5py


def PCA(input_data,red_size): # this function alone returns 84 values and original and reconstructed images also 
    labels=np.array(input_data['labels_train'])
    digits=np.array(input_data['digits_train'])
    
    mean_data=[]                                                        # data centred wrt to mean
    cov_matrix=[]                                                       # array of cov_matrices  
    mean_digit=np.zeros((10,784),dtype=float)                           # array of mean vectors
    digit_count=[0,0,0,0,0,0,0,0,0,0]
    data=[]                                                             # complete data not wrt to mean  
    resize_input=[]                                                 # resizing input with 784 size
    for i in range(0,60000):
        resize_input.append(digits[i].reshape(784,order='F'))


    for j in range(0,10):
        row=[]
        for i in range(0,len(labels[0])):
            if(labels[0][i]==j):
                mean_digit[j]=mean_digit[j]+resize_input[i]
                row.append(resize_input[i])
                digit_count[j]=digit_count[j]+1                        # increasing digit count 
        mean_digit[j]=mean_digit[j]/float(digit_count[j])
        data.append(row)
        mean_data.append(data[j]-mean_digit[j])
        cov_matrix.append(np.cov(mean_data[j],rowvar = False))

    for digit in range(0,10):                                         # displaying all mean_data for each digit 
        temp_img = mean_digit[digit].reshape(28,28)
        plt.imshow(temp_img,cmap="turbo")
        plt.title(f"q5_{digit}_mean_img.jpg")
        plt.show()

   
    # caluculating the 84 eigen vectors and eigen values 
    for i in range(0,10):
        eigen_values , eigen_vectors = np.linalg.eigh(cov_matrix[i])
        sorted_index = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_index]
        eigen_vectors = eigen_vectors[:,sorted_index]
        eigen_vectors = eigen_vectors[:,0:red_size]
        img_1 = data[i][0].reshape(28,28)
        plt.imshow(img_1,cmap="gray")
        plt.title(f"q5_{i}_original_img.jpg")
        plt.show()
        img = data[i][0] - mean_digit[i]
        score = np.dot(eigen_vectors.transpose(),img)
        constructed_img = np.dot(eigen_vectors,score)+mean_digit[i]
        constructed_img = np.uint8(np.absolute(constructed_img))
        a = constructed_img.reshape(28,28)
        plt.imshow(a,cmap="gray")
        plt.title(f"q5_{i}_reconstructed_img.jpg")
        plt.show()
        return score
    



in_data = h5py.File('../data/mnist.mat','r')
labels = PCA(in_data,84) # this labels will return the 84 values 