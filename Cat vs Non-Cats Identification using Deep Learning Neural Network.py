#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IT16001480 - H.K.D.C.Jayalath
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# In[ ]:


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

filename = 'train_catvnoncat.h5'
f = h5py.File(filename, 'r')
list(f.keys())
data = list(f['train_set_x'])


# In[ ]:


# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It is a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# In[ ]:




import matplotlib.pyplot as plt
for i in range(50,55):
    print(data[i].shape)
    plt.imshow(data[i])
    plt.show()


# In[ ]:


# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# In[ ]:


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# In[ ]:



def normalize(x):
    """
        argument
            - x: input image data in numpy array [64, 64, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


# In[ ]:



y_tr= np.array(list(f['train_set_y']))


# In[ ]:



import seaborn as sns
sns.set_style('darkgrid')
sns.countplot(y_tr,palette='twilight')


# $12,288$ equals $64 \times 64 \times 3$ which is the size of one reshaped image vector.

# In[ ]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# In[ ]:


# GRADED FUNCTION: 4_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
    np.random.seed(1)
    costs = []   # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X,parameters)
        # Compute cost.
        cost = compute_cost(AL,Y)
        # Backward propagation.
        grads = L_model_backward(AL,Y,caches)
        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)
       
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)         
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


# In[ ]:


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[ ]:


print("train_x shape = "+str(train_x.shape))
print("train_y shape = "+str(train_y.shape))
pred_train = predict(train_x, train_y, parameters)


# In[ ]:


print("test_x shape = "+str(test_x.shape))
print("test_y shape = "+str(test_y.shape))
pred_test = predict(test_x, test_y, parameters)


# 6) Results Analysis
#  

# In[ ]:


print_mislabeled_images(classes, test_x, test_y, pred_test)


# 7) Test with your own image
# 
# 

# In[ ]:


## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

