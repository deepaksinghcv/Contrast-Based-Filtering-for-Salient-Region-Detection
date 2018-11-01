#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np


# In[12]:


def abstract(i):  
    
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    #image = cv2.imread('flower.jpg')
    image=i
    image_norm=img_as_float(image)
    lab = rgb2lab(image_norm)
    lab_norm=(lab + np.array([0, 128, 128])) / np.array([100, 255, 255])
    #image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    # loop over the number of segments
    segments = slic(image_norm, n_segments = 600, compactness=15, sigma = 1,multichannel=True, convert2lab=True)

    # show the output of SLIC

    plt.imshow(mark_boundaries(image, segments))
    plt.savefig('segment.jpg')
    plt.show()
    n_segments = segments.max() + 1

    # construct position matrix
    max_y, max_x = np.array(segments.shape) - 1
    x = np.linspace(0, max_x, image.shape[1]) / max_x
    y = np.linspace(0, max_y, image.shape[0]) / max_y
    position = np.dstack((np.meshgrid(x, y)))

    # mean_lab = np.zeros((n_segments, 3))
    # mean_rgb = np.zeros((n_segments, 3))
    # mean_position = np.zeros((n_segments, 2))
    # for superpixel in np.unique(superpixels):
    #     mask = superpixels == superpixel
    #     mean_lab[superpixel, :] = nlab[mask, :].mean(axis=0)
    #     mean_rgb[superpixel, :] = nrgb[mask, :].mean(axis=0)
    #     mean_position[superpixel, :] = position[mask, :].mean(axis=0)
    mean_colors = np.zeros((n_segments, 3))
    mean_position = np.zeros((n_segments, 2))
    clrs=np.zeros((n_segments, 3))
    for i in np.unique(segments):
        mask = segments == i
        mean_colors[i,:]=lab[mask,:].mean(axis=0)
        mean_position[i, :] = position[mask, :].mean(axis=0)

    mean_position
    mean_colors
    d_abstract={}
#     for j in range (mean_position.shape[0]):
#         d_abstract[j]=[]

#     for k in range (mean_position.shape[0]):

#         d_abstract[k].append([[mean_position[i]],[mean_colors[i]]])

    d={}
    d1={}
    d_abst={}
    d_uniqueness={}
    r,c=segments.shape
    for i in range (r):
        for j in range (c):
            d[segments[i,j]]=[]
            d_uniqueness[segments[i,j]]=[]
            d1[segments[i,j]]=[]


    for i in range (r):
        for j in range (c):
            d[segments[i,j]].append([i,j])
            d1[segments[i,j]].append(image_norm[i,j])

    image_copy=np.copy(image_norm)
    
    for i in range (r):
        for j in range (c):
            image_copy[i,j]= np.median(d1[segments[i,j]],axis=0)
            d_uniqueness[segments[i,j]].append([image_copy[i,j]])
            
            
    output=(image_copy*255).astype('uint8')
    for i in d_uniqueness.keys():
        
        clrs[i]=d_uniqueness[i][0][0]

    
    return clrs,mean_position,segments,d,output,d_uniqueness

    
    
    


# In[5]:


# d={}
# d_abst={}
# d_uniqueness={}
# r,c=segments.shape
# for i in range (r):
#     for j in range (c):
#         d[segments[i,j]]=[]
#         d_uniqueness[segments[i,j]]=[]
#         d_abst[segments[i,j]]=[]
        

# for i in range (r):
#     for j in range (c):
#         d[segments[i,j]].append([i,j])
        
# len(d)


# In[6]:


# import numpy as np
#image_copy=np.copy(image_norm)
# for i in range (r):
#     for j in range (c):
#         image_copy[i,j]= np.median(d[segments[i,j]],axis=0)
#         d_uniqueness[segments[i,j]].append([[i,j] ,[image_copy[i,j]]])
# cv2.imwrite('image_copy.jpg',(image_copy * 255).astype('uint8'))


# In[13]:


from skimage import io
image=io.imread('flower.jpg')
image_uniqueness=np.zeros((image.shape[0],image.shape[1],3))
image_distribution=np.zeros((image.shape[0],image.shape[1],3))

colors=[]
positions=[]
colors,positions,seg,d_p,o,d_u = abstract(image)
io.imshow(o)
io.show()
#d_u


# In[14]:



sigmap = 1
sigmac = 30


def giveSSD(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

def wp(pi,pj):
    Zi = 1 #change this
    return (1/Zi)*np.exp((-1/2*(sigmap**2))*np.square(giveSSD(pi,pj)))

def wc(ci,cj):
    Zi = 18
    #change this
    return (1/Zi)*np.exp((-1/2*(sigmac**2))*np.square(giveSSD(ci,cj)))

def uniquenessAssignment(c,p):
    U = np.empty(len(c))
    for i in range(len(c)):
        pi = p[i]
        ci = c[i]
        tUniq = 0;
        for j in range(len(c)):
            #if i != j:
            pj = p[j] 
            cj = c[j]
            tUniq += np.square(giveSSD(ci,cj))*wp(pi,pj)

        U[i] = tUniq
    
    return U

def distributionAssignment(c,p):
    D = np.zeros(len(c))
    for i in range(len(c)):
        
        ci = c[i]
        pi = p[i]
        
        meani = 0
        for j in range(len(c)):
            cj = c[j]
            pj = p[j]
            #if (i != j):
            meani += wc(ci,cj)*pj
        
        tDist = 0
        for j in range(len(c)):
            cj = c[j]
            pj = p[j]
            #if( i != j):
            tDist += np.square(giveSSD(pj,meani))*wc(ci,cj)
            
        D[i] = tDist
        
    return D


# In[15]:


Uniqueness=uniquenessAssignment(colors,positions)
Uniqueness.shape


# In[16]:


dist=distributionAssignment(colors,positions)
dist.shape


# In[20]:


im_test=[]

for i in range(len(d_p)):
    for k in range(len(d_p[i])):
        
        row=d_p[i][k][0]
        col=d_p[i][k][1]
        image_uniqueness[row,col]=Uniqueness[i]
        image_distribution[row,col]=dist[i]
io.imshow(image_uniqueness/255,cmap='gray')
io.show()
io.imshow(255-np.uint8(image_distribution*255))
io.show()
cv2.imwrite('dis.jpg',255-np.uint8(image_distribution*255))
cv2.imwrite('un.jpg',image_uniqueness)


# In[ ]:


wp(colors[0],colors[1])


# In[ ]:


dist

