#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
# Importing required libraries

import os
from sklearn.decomposition import PCA
os.chdir('C:/Users/D3V1L/Pictures/DITF')
# Loading the image 
img = imread('21.jpeg') #you can use any image you want.
plt.imshow(img)



# In[12]:


blue,green,red = cv2.split(img)

#initialize PCA with first 20 principal components
pca = PCA(20)
 
#Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
 
#Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
 
#Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)
img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
#viewing the compressed image
plt.imshow(img_compressed)


# In[13]:


#initialize PCA with first 20 principal components
pca = PCA(150)
 
#Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
 
#Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
 
#Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)
img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
#viewing the compressed image
plt.imshow(img_compressed)


# In[ ]:




