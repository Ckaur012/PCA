#!/usr/bin/env python
# coding: utf-8

# # PCA: Image Dataset
# 
# We will first split the image into the three channels (Blue, Green, and Red) first and then and perform PCA separately on each dataset representing each channel and will then merge them to reconstruct the compressed image. Hence, if our colored image is of shape (m, n, 3), where (m X n) is the total number of pixels of the image on the three channels (b, g, r
# 
# We can also perform the same thing without splitting into blue, green, and red channels and reshaping the data into (m, n X 3) pixels, but we have found that the explained variance ratio given by the same number of PCA component is better if we use the splitting method as mentioned in the earlier paragraph.
# 
# I will use the following photograph for the demonstration.

# ![1*Uf9xZsakLqYww5LgsapjrQ.png](attachment:1*Uf9xZsakLqYww5LgsapjrQ.png)

# # Load and pre-process the image

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from scipy.stats import stats
import matplotlib.image as mpimg


# Now let’s read the image rose.jpg and display it.
# 

# In[2]:


img = cv2.cvtColor(cv2.imread('/Users/charanjeetkaur/Downloads/1*Uf9xZsakLqYww5LgsapjrQ.png'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# Check the shape using the following line:

# In[3]:


img.shape


# Now, I will split the image into 3 channels and display each image:

# In[4]:


#Splitting into channels
blue,green,red = cv2.split(img)
# Plotting the images
fig = plt.figure(figsize = (15, 7.2)) 
fig.add_subplot(131)
plt.title("Blue Channel")
plt.imshow(blue)
fig.add_subplot(132)
plt.title("Green Channel")
plt.imshow(green)
fig.add_subplot(133)
plt.title("Red Channel")
plt.imshow(red)
plt.show()


# Let’s verify the data of the blue channel:

# In[5]:


blue_temp_df = pd.DataFrame(data = blue)
blue_temp_df


# I will divide all the data of all channels by 255 so that the data is scaled between 0 and 1.

# In[6]:


df_blue = blue/255
df_green = green/255
df_red = red/255


# In[7]:


df_blue
df_green
df_red


# # Fit and transform the data in PCA
# We already have seen that each channel has 485 dimensions, and we will now consider only 50 dimensions for PCA and fit and transform the data and check how much variance is explained after reducing data to 50 dimensions.

# In[11]:


pca_b = PCA(n_components=50)
pca_b.fit(df_blue)
trans_pca_b = pca_b.transform(df_blue)


# In[9]:


pca_g = PCA(n_components=50)
pca_g.fit(df_green)
trans_pca_g = pca_g.transform(df_green)


# In[10]:


pca_r = PCA(n_components=50)
pca_r.fit(df_red)
trans_pca_r = pca_r.transform(df_red)


# We have fitted the data in PCA, let’s check the shape of the transformed image of each channel:

# In[12]:


print(trans_pca_b.shape)
print(trans_pca_r.shape)
print(trans_pca_g.shape)


# Let’s check the sum of explained variance ratios of the 50 PCA components (i.e. most dominated 50 Eigenvalues) for each channel.

# In[13]:


print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")


# Finally we get 99% of variance in data which is perfect

# Let's plot bar charts to check the explained variance ratio by each Eigenvalues separately for each of the 3 channels:

# In[15]:


ig = plt.figure(figsize = (15, 7.2)) 
fig.add_subplot(131)
plt.title("Blue Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1,51)),pca_b.explained_variance_ratio_)
plt.show()


# In[16]:


fig.add_subplot(132)
plt.title("Green Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1,51)),pca_g.explained_variance_ratio_)
plt.show()


# In[17]:


fig.add_subplot(133)
plt.title("Red Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1,51)),pca_r.explained_variance_ratio_)
plt.show()


# # Reconstruct the image and visualize
# We have completed our PCA dimensionality reduction. Now we will visualize the image again and for that, we have to reverse transform the data first and then merge the data of all the 3 channels into one.

# In[18]:


b_arr = pca_b.inverse_transform(trans_pca_b)
g_arr = pca_g.inverse_transform(trans_pca_g)
r_arr = pca_r.inverse_transform(trans_pca_r)
print(b_arr.shape, g_arr.shape, r_arr.shape)


# We will merge all the channels into one and print the final shape:

# In[19]:


img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
print(img_reduced.shape)


# The exact shape of the original image that we had imported at the very beginning. Now we will display both the Images (original and reduced) side by side.

# In[20]:


fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(img)
fig.add_subplot(122)
plt.title("Reduced Image")
plt.imshow(img_reduced)
plt.show()


# It's amazing to see that the compressed image is very similar (at least we can still identify it as a rose) to that of the original one although we have reduced the dimension individually for each channel to only 50 from 485. But, we have achieved our goal. No doubt that now the reduced image will be processed much faster by the computer.

# In[ ]:




