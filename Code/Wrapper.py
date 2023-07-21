#!/usr/bin/env python


import numpy as np
import cv2 as cv
from helper_functions import *
import matplotlib.pyplot as plt
import argparse 
import math as m
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' , '--Imagename' , default = '1.jpg')
    Args = parser.parse_args()
    name = Args.Imagename
    # img = cv.imread('/home/sheriarty/pdave1_hw0/Phase1/BSDS500/Images/' + str(name))
    img = cv.imread('/..BSDS500/Images/' + str(name))
    imgray = img[:,:,0]
    
    size = 49
    shape = [1,3]
    orientations = 8
    dog = orientedDoGFilter(size, shape,orientations)
    filters = dog.shape[2]
    fig1 = plt.figure()
    
    for i in range(filters):
        ax = fig1.add_subplot(2,8,i+1)
        plt.imshow(dog[:,:,i] , 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig1.suptitle("Deravitive of Gaussian FilterBank" , fontsize = 14 , fontweight = 'bold')  
    plt.savefig('../Outputs/DoG.png')
    
    
    size = 49
    lm = LMFilters(size)
    
    filters = lm.shape[2]
    fig1 = plt.figure()
    for i in range(filters):
        ax = fig1.add_subplot(4,12,i+1)
        plt.imshow(lm[:,:,i] , 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig1.suptitle("LM FilterBank", fontsize = 14 , fontweight = 'bold')
    plt.savefig('../Outputs/LM.png')
    
    params = [[3 , 1.9 *np.pi,1],[6,2.5*np.pi,1],[9,3.9*np.pi,1],[12,4.5*np.pi,0.7],[15,6*np.pi,0.7]]
    
    size = 49
    
    orientations = 8
    gb = gaborfilterbank(size , params, orientations)
    filters = gb.shape[2]
    fig1 = plt.figure()
    for i in range(filters):
        ax = fig1.add_subplot(5,8,i+1)
        plt.imshow(gb[:,:,i] , 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig1.suptitle("Gabor Filter Bank" , fontsize = 14 , fontweight = 'bold')
    plt.savefig('../Outputs/GB.png')
    print("Filter Bank generation complete")
    
    
    half = circularKernel(49)
    orientation = [10,15,25]
    d1 , d2 , d3 = halfmaskdisc(8,15, 25 ,16)
    
    
    flt = list()
    for i in range(d1.shape[2]):
        flt.append(d1[:,:,i])
    for i in range(d2.shape[2]):
        flt.append(d2[:,:,i])
    for i in range(d3.shape[2]):
        flt.append(d3[:,:,i])
        
    fig1 = plt.figure()
    
    for i in range(len(flt)):
        ax = fig1.add_subplot(6,8,i+1)
        plt.axis = 'off'
        plt.imshow(flt[i] , cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig1.suptitle("Halfmask Discs" , fontsize = 14 , fontweight = 'bold')
    plt.savefig('../Outputs/HalfmaskDiscs.png')
    print("Half-mask Disc generation complete")
    
    dog_resp  = filterResponse(imgray , dog)
    log_resp = filterResponse(imgray , lm)
    lm_resp = filterResponse(imgray , gb)
    
    filterresponse = np.dstack((dog_resp , log_resp, lm_resp ))
    # texton_map = texton(imgray , dog , lm,gb , 64)
    texton_map = do_kmeans(filterresponse , 64)
    # texton_map = do_kmeans(dog_resp , 64)
    fig = plt.figure()
    plt.imshow(texton_map)
    namesplot = name.split('.')
    fig.suptitle("Texton Map" , fontsize = 20)
    plt.savefig("../Outputs/TextonMap_" + namesplot[0] + ".png")
    print("Texton Map generation complete")
   
    
    
    bins = 64
    diskbanks = [d1 , d2 , d3]
    pairs = 8
    
    tg = gradient(texton_map , bins , diskbanks , pairs)
    fig1 = plt.figure()
    plt.imshow(tg)
    namesplot = name.split('.')
    filename = "../Outputs/Tg_" + namesplot[0] + ".png"
    fig1.suptitle("Texton Gradient" , fontsize = 20)
    plt.savefig(filename)
    
        
    print("Done T_g")
    k = 16
    brightnessmap = do_kmeans(imgray , k)
    fig1 = plt.figure()
    plt.imshow(brightnessmap)
    filename = "../Outputs/B_Map_" + namesplot[0] + ".png"
    fig.suptitle("Brightness Map" , fontsize = 20)
    plt.savefig(filename)
    
    
    
    
    bg = gradient(brightnessmap , bins , diskbanks , pairs)
    fig = plt.figure()
    plt.imshow(bg)
    filename = "../Outputs/Bg_" + namesplot[0] + ".png"
    fig.suptitle("Brightness Gradient" , fontsize = 14 , fontweight = 'bold')
    plt.savefig(filename)
    print("Brightness Gradient generation complete")
    
    
    
    colormap = do_kmeans(img , k)
    fig = plt.figure()
    plt.imshow(colormap)
    filename = "../Outputs/C_Map_" + namesplot[0] + ".png"
    fig.suptitle("Color Map" , fontsize = 14 , fontweight = 'bold')
    plt.savefig(filename)
    print("Color Gradient generation complete")
    
    
    cg = gradient(colormap , bins , diskbanks , pairs)
    fig = plt.figure()
    plt.imshow(cg)
    namesplit = name.split('.')
    
    filename = "../Outputs/Cg_" + namesplit[0] + ".png"
    fig.suptitle("Color Gradient" , fontsize = 14 , fontweight = 'bold')
    plt.savefig(filename)
    # cv.imwrite("../Outputs/CMap_" + namesplot[0] + ".png" , colormap)
    print("Done C_g")
    
    
    # sobel = cv.imread("/home/sheriarty/pdave1_hw0/Phase1/BSDS500/SobelBaseline/" + namesplit[0] + ".png",0)
    # canny = cv.imread("/home/sheriarty/pdave1_hw0/Phase1/BSDS500/CannyBaseline/" + namesplit[0] + ".png",0)
    sobel = cv.imread('../BSDS500/SobelBaseline/' + namesplit[0] + ".png",0)
    canny = cv.imread('../BSDS500/CannyBaseline/' + namesplit[0] + ".png",0)
    a = (tg + bg + cg)/3
    w = 0.2
    b = w*canny + (1-w)*sobel
    pb = np.multiply(a,b)
    cv.normalize(pb,pb,0,255,cv.NORM_MINMAX)
    
    fig1 = plt.figure()
    plt.imshow(pb)
    
    filename = "../Outputs/PbLite" + namesplit[0] + ".png"
    fig.suptitle("Pblite Result" , fontsize = 14)
    
    cv.imwrite("../Outputs/PbLite" + namesplit[0] + ".png" , pb)
    print("Pb-lite generation complete")
if __name__ == '__main__':
    main()
