import cv2 as cv
import numpy as np
import math as m
import scipy
import scipy.stats as st
import skimage.transform
import argparse
import imutils
import sklearn.cluster 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# import scipy.stats as st


#Calcutates DoG filterbank
def gaussianKernal2D(size , sigma):
    if(size%2) != 1:
        size = size+1
    size = int(size) //2
    
    x , y  = np.mgrid[-size:size+1 , -size:size+1]
    div = np.sqrt(2 * np.pi * sigma**2)
    
    gau = np.exp(-(x**2 + y**2 / 2.0 * sigma**2))/div
    
    return gau/gau.sum()
    
def mvGaussian(size , u1 = 0 , u2 = 0 , sigma1 = 1 , sigma2 = 3):
    gau = np.zeros((size,size), dtype = np.float32)
    if(size%2) != 1:
        size = size+1
    dim = int(size) // 2
    
    x = np.arange(-dim,dim+1)
    y = np.arange(-dim,dim+1)
    
    for i in x:
        for j in y:
            G1 = np.exp(-((i-u1)**2)/(2*(sigma1**2)))
            G2 = np.exp(-((j-u2)**2)/(2*(sigma2**2)))
            X = i + dim
            Y = j + dim
            gau[X,Y] = G1*G2
    return gau

def orientedDoGFilter(size = 9 , scale = [1,2] , orientation = 8):
    count = 0
    filter_count = len(scale)*orientation
    DoGfilterBank = np.zeros((size,size , filter_count ), dtype = np.float32)
    
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    
    for s in scale:
        gaussian_kernel = gaussianKernal2D(size , sigma = s)
         
        DoG = cv.filter2D(gaussian_kernel , -1 , sobel)    
        rotation = 360/orientation
        
        for i in range(orientation*len(scale)):
            angle = i*rotation
            DoGfilterBank[:,:,count]  = imutils.rotate(DoG, angle)
            count += 1
            
        return DoGfilterBank
    
#Calculate LM DoG filters
def LMDoGFilter(size = 49 ,scale = [1 , np.sqrt(2) , 2] , orientations = 6):
    
    index = 0
    
    sobel_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = np.float32)
    
    gau1d = np.zeros((size,size , len(scale)*orientations) , dtype = np.float32)
    gau2d = np.zeros((size , size , len(scale)*orientations) , dtype = np.float32)
    
    for i in scale:
        mvker  = mvGaussian(size , u1 = 0 , u2= 0 , sigma1 = 1*i , sigma2 = 3*i)
        dog1 = cv.filter2D(mvker  , -1 , sobel_kernel.T)
        dog2 = cv.filter2D(dog1 , -1 , sobel_kernel.T)
        
        orientation = 180/orientations
        
        for j in range(orientations):
            angel = j*orientations
            gau1d[: , : , index] = imutils.rotate(dog1 , angel)
            gau2d[: , : , index]  = imutils.rotate(dog2 , angel)
            
            index += 1
    
    DOGfilters = np.dstack((gau1d , gau2d))
    return DOGfilters

#Calculate LM LoG filters

def LMLoGFilters(size = 49 , scale = [1 , np.sqrt(2) , 2 , 2*np.sqrt(2) , 3 , 3*np.sqrt(2) , 6 , 6*np.sqrt(2) ]): 
    var = scale*scale
    shape  = (size , size)
    m  , _ = [(i-1)/2 for i in shape]
    x , y  = np.ogrid[-m:m+1 , -m:m+1]
    gau = (1/np.sqrt(2*np.pi*var))*np.exp(-(x*x + y*y) / (2*var))
    h = gau* ((x*x + y*y) - var)/(var*var) 
    return h 

#Calculate LM Gau Filters
def LMGauFilters(size = 49 , scale = [1 , np.sqrt(2) , 2 , 2*np.sqrt(2)]):
    gauFilter = np.zeros((size , size , len(scale)), dtype = np.float32)
    for i,j in enumerate(scale):
        gauFilter[:,:,i] = gaussianKernal2D(size , sigma = j)
    return gauFilter

#Calculate LM FilterBank

def LMFilters(size = 49):
    scale = [1 , np.sqrt(2) , 2 ]
    
    gauder = LMDoGFilter(size , scale , orientations = 6)
    
    F = np.zeros((size,size,8),dtype = np.float32)
    scales = [1 , np.sqrt(2) , 2 , 2*np.sqrt(2)]
    index = 0
    
    for i in range(len(scales)):
        F[:,:,index] = LMLoGFilters(size , scales[i])
        index += 1    
        
    for i in range(len(scale)):
        F[: , : , index] = LMLoGFilters(size , 3*scale[i])
        index += 1 
    gaufilter = LMGauFilters(size , scales)
    return np.dstack((gauder , F , gaufilter ))
#     return gauder , F , gaufilter


#Calculate gaborfilters    
def gaborfilter(size , sigma, theta, lamda , gamma , psi):
    m = size//2
    x,y = np.mgrid[-m:m+1 , -m:m+1]
    x_theta = x* np.cos(theta) + y *np.sin(theta)
    y_theta = -x * np.sin(theta) + y *np.cos(theta)
    
    g = np.exp((-0.5/sigma**2) * (x_theta**2 + gamma**2 *y_theta**2)) * np.cos(2*np.pi/lamda * x_theta + psi)
    return g
    
def gaborfilterbank(size,parameters,orientations = 8):
    angel = 180/orientations
    gbfilterbank = np.zeros((size,size,len(parameters)*orientations),dtype = np.float32)
    
    index = 0
    for i in parameters:
        for j in range(orientations):
            theta = -angel*j
            gbfilterbank[:,:,index] = gaborfilter(size = size, sigma = i[0],theta = theta , lamda = i[1],gamma = i[2],psi = 0)
#             print(index,j)
            index += 1
            
    return gbfilterbank

# Computing texton map using image and filter bank
def filterResponse(img , filterbank):
    filter_count = filterbank.shape[2]
    w , h  = img.shape
    
    texton = np.zeros((w,h,filter_count) , dtype=np.float32)
    
    for i in range(filter_count):
        texton[:,:,i] = cv.filter2D(img , -1 , filterbank[:,:,i])
    return texton
        
# old implementation of texton map using 3 different filter banks
def texton(img,fb1,fb2,fb3 , clusters):
    w , h  = img.shape
    texton_DOG = filterResponse(img , fb1) 
    texton_LM = filterResponse(img ,fb2)
    texton_Gabor = filterResponse(img , fb3)
    
    tex_map = np.dstack((texton_DOG[:,:,1:], texton_LM[:,:,1:], texton_Gabor[:,:,1:]))
    
    a , b  ,r = tex_map.shape
    res = np.reshape(tex_map,((w*h),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = clusters , random_state = 0)
    kmeans.fit(res)
    labels = kmeans.predict(res)
    tex = np.reshape(labels , (a , b))
    
    return tex    
# created texton map efficiently
def getTexton(img, fb , cluster):
    filter = fb.shape[2]
    w , h = img.shape
    fbresponce = np.zeros((w , h , filter) , dtype = np.float32)
    for i in range(filter):
        fbresponce[:,:,i] = cv.filter2D(img , -1 , fb[:,:,i])
        
    responce = fbresponce.reshape((-1,filter)).astype(np.float32)  
    kmeans = KMeans(n_clusters = cluster , random_state = 0).fit(responce)  
    labels = kmeans.labels
    tmap = labels.reshape(w , h)
    return tmap

#does kmeans for a image given k clusters
def do_kmeans(fr , k):
    if len(fr.shape)>2:
        w , h , filters = fr.shape
    else:
        w , h = fr.shape
        filters = 1 
    flat = fr.reshape((-1,filters)).astype(np.float32)
    kmeans = KMeans(n_clusters = k ,random_state = 0).fit(flat)
    labels = kmeans.labels_
    tmap = labels.reshape(w,h)
    return tmap 

#Calculate Brightness Map           
def brightness(img , clusters):
    a , b   = img.shape
    img = np.reshape(img,((a*b) , 1))
    kmeans = sklearn.cluster.KMeans(n_clusters = clusters , random_state = 2)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    l = np.reshape(labels,(a,b))
    plt.imshow(l,cmap = 'binary')
    
    return l 


      

# def gradient(img , bins , filter_bank):
#Calculate halfmask discs
def circularKernel(size):
    radius = size//2
    
    center = (radius,radius)
    half = np.zeros((size,size),dtype = np.float32)
    cv.circle(half , center , radius , color = 255 , thickness = -1)
    half[:size//2+1] = 0
    return half

def halfmaskdisc(a,b,c,orientation):
    mask1 = circularKernel(a)
    mask2 = circularKernel(b)
    mask3 = circularKernel(c)
    angle = 360 / orientation
    
    halfdiskbank1 = np.zeros((a,a,orientation),dtype = np.float32)
    halfdiskbank2 = np.zeros((b,b,orientation),dtype = np.float32)
    halfdiskbank3 = np.zeros((c,c,orientation),dtype = np.float32)
    
    for i in range(orientation):
        degree = angle*i
        halfdiskbank1[:,:,i]  = imutils.rotate(mask1,degree)
        halfdiskbank2[:,:,i]  = imutils.rotate(mask2,degree)
        halfdiskbank3[:,:,i]  = imutils.rotate(mask3,degree)
        
    return halfdiskbank1, halfdiskbank2 , halfdiskbank3


#Calculate chi_sqr 
def chi_sqr(img , num , filter1 , filter2):
    chi_sqr_dist = img*0
    for i in num:
        g = cv.filter2D(img , -1 , filter1)
        h = cv.filter2D(img , -1 , filter2)
        chi = (chi_sqr_dist + ((g-h)**2)/g+h)/2
    return chi
#Calculate final gradient given image , bins , filter bank and diskpairs
def gradient(img , bins ,fb , diskpairs):
    filters = len(fb) * diskpairs
    gradi = np.zeros((img.shape[0] , img.shape[1] , filters) , dtype = np.float32)
    
    index = 0
    dummyarr = np.ones((img.shape),dtype=np.float32)*0.0000001    
    for disk in fb:
        rightdisks = disk[:,:,:diskpairs]
        leftdisks = disk[:,:,diskpairs:]
        if leftdisks.shape[2] == rightdisks.shape[2]:
            for d in range(rightdisks.shape[2]):
                rightmask = rightdisks[:,:,d]
                leftmask = leftdisks[:,:,d]
                chi = np.zeros((img.shape[0] , img.shape[1]),dtype = np.float32)
            
                
                for i in range(bins):
                    tmap = img.copy()
                    tmap[img == i] = 1.0
                    tmap[img !=i] = 0.0
                    g = cv.filter2D(tmap , -1 , leftmask)
                    h = cv.filter2D(tmap , -1 , rightmask)
                    chi = chi + ((g - h)**2)/(1/(g+h + dummyarr))
                    
                gradi[:,:,index] = chi*0.5
                index += 1
    grad  = gradi.mean(axis = 2 , dtype = np.float32)
    
    return grad

