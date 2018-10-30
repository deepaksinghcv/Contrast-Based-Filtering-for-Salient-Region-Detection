#!/usr/bin/env python
# coding: utf-8

# In[135]:


# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
 
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = io.imread("flower.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
# loop over the number of segments
segments = slic(image, n_segments = 500, compactness=12, sigma = 1,multichannel=True, convert2lab=True)

# show the output of SLIC

fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.savefig('segment.jpg')
plt.show()
segments


# In[141]:


segments


# In[132]:



d={}
d_uniqueness={}
r,c=segments.shape
for i in range (r):
    for j in range (c):
        d[segments[i,j]]=[]
        d_uniqueness[segments[i,j]]=[]

for i in range (r):
    for j in range (c):
        d[segments[i,j]].append(image[i,j])
        


# In[134]:


import numpy as np
image_copy=np.copy(image)
for i in range (r):
    for j in range (c):
        image_copy[i,j]= np.median(d[segments[i,j]],axis=0)
        d_uniqueness[segments[i,j]].append([[i,j] ,[image_copy[i,j]]])
cv2.imwrite('image_copy.jpg',image_copy)
image_copy
d_uniqueness


# In[128]:


d={}
for i in range (r):
    for j in range (c):
        d[segments[i,j]]=[]
        d_uniqueness[segments[i,j]]=[]
for i in range (r):
    for j in range (c):
        d[segments[i,j]].append([[i,j],[image[i,j]]])


# In[123]:


d_uniqueness

