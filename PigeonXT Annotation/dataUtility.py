# python3 dataUtility.py -h
# python3 dataUtility.py -img /Users/sabyasachi/Documents/images_v1 -pAS True -pSS True
# if i == 4: break
## search the above and remove it from the code

import os 
import pickle
import re
import sys
import argparse
from argparse import ArgumentParser
import shutil
import compareFolders as cf
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2



"""
        -- This module is dedicated to image data manipulation for annotation usingh pigeonXT
            - take in image and target directory
            - match them and divide them into parts 
            - save the parts into a new directory 
            - pass the part through pigeonXT

        -- Flow:
            - image/Mask Directory --> combines image and mask getting finalImage
            - 
"""

def parseArgs():
    # https://realpython.com/command-line-interfaces-python-argparse/
    parser = argparse.ArgumentParser(description="something")
    parser.add_argument('-img', "--imageDirectory", help="Input directory of the image and target files", 
                        default = '/Users/sabyasachi/Documents/images_v1') #os.getcwd(), type = str)
    parser.add_argument('-pAS', "--plotAllSpheroids", help="If you want to plot all the spheroids", default = False)
    parser.add_argument('-pSS', "--plotSingleSpheroid", help="If you want to plot all the spheroids", default = False)
    parser.add_argument('-dev', "--develop", help="Develpoment mode or not", action = 'store_true')  

    args = parser.parse_args()
    return args



# def divideImage(image):
#     return imageFolder

def fullLabelledImage(image, mask, numLabels, labels, 
					  stats, centroids, plotAllSpheroids = False):
	# loop over the number of unique connected component labels
	for i in range(0, numLabels):
		if i == 0:
			text = "examining component {}/{} (background)".format(i , numLabels)
			continue
		else:
			# extract the connected component statistics and centroid for the current label
			x = stats[i, cv2.CC_STAT_LEFT]
			y = stats[i, cv2.CC_STAT_TOP]
			w = stats[i, cv2.CC_STAT_WIDTH]
			h = stats[i, cv2.CC_STAT_HEIGHT]
			(cX, cY) = centroids[i]
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(image, str(i), (int(cX), int(cY)), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
			# cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), i) #add the centroid to the image
	if plotAllSpheroids == True:		
		plt.figure(figsize=(10,10))
		plt.imshow(image)
		plt.title(f"Total Embryonic Bodies = {numLabels-1}", fontsize = 10)
		plt.show()
	return image

	
def singleSpheroidImage(name, image, mask,
                                numLabels, labels, stats, centroids, plotSingleSpheroid = False):
            
            spheroidInfoDic = {}
            # loop over the number of unique connected component labels
            for i in range(0, numLabels):
                  if i == 0:
                        text = "examining component {}/{} (background)".format(i , numLabels-1)
                        # print("[INFO] {}".format(text))
                        continue
                  else:
                        text = "examining component {}/{}".format( i, numLabels-1)
                        # print("[INFO] {}".format(text))

                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        area = stats[i, cv2.CC_STAT_AREA]
                        (cX, cY) = centroids[i]
                        spheroidName = str(f'spheroid{i}_' + name)
                        # cv2.putText(image, str(i), (int(cX), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # adding text to the image
                        croppedImage = np.array(image[y:y+h, x:x+w, :])
                        componentMask = (labels == i).astype("uint8") * 255 # construct a mask for the current connected component and dilate it to increse teh boundary
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                        componentMask = cv2.dilate(componentMask, kernel, iterations=2)
                        croppedImage = np.where(componentMask[y:y+h, x:x+w, None], croppedImage, 0)
                        # spheroidList.append(croppedImage)
                        spheroidInfoDic[spheroidName] = [area, centroids[i], croppedImage]

                        if plotSingleSpheroid == True:
                                plt.figure(figsize=(5,5))
                                plt.imshow(croppedImage)
                                plt.title(f"area = {area}; \nSpheroid number = {i}", fontsize = 10)
                                plt.show()
            return spheroidInfoDic
     

def divideAndLabelSpheroids(name, image, mask, plotAllSpheroids = False, plotSingleSpheroid = False):
      """
      Takes in the image and mask and returns the image with the labelled spheroids
      """
      output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
      (numLabels, labels, stats, centroids) = output
      
      imgCopy = image.copy()
      ## display this one in a separate cell for user to get the whole picture and make plotAllSpheroids = True
      labelledSpheroids = fullLabelledImage(imgCopy, mask,
									    numLabels, labels, stats, centroids, plotAllSpheroids)
      ## two dictionary numSpheroids{} for annotation and unNumSpheroids{} for further downstreaming analysis and all channels
      spheroidInfoDic = singleSpheroidImage(name, image, mask, 
									numLabels, labels, stats, centroids, plotSingleSpheroid)
      return labelledSpheroids, spheroidInfoDic 



def combineInputMask(inputImageList, inputDir, 
                     targetMaskList, targetDir, develop):
    """
    Takes in the list of the image and mask files
    and returns the array of finallyCombinedImage and the mask with the name of the image
    """
    finalCombineImage_InfoDic = {}

    for i, name in enumerate(inputImageList):
        if name == '.DS_Store':continue
        mask = name.split('.')[0] + '_bn.tif'
        if mask in targetMaskList:
            img = np.array(Image.open(os.path.join(inputDir, name)))
            mask = np.array(Image.open(os.path.join(targetDir, mask)))
            finalImage = np.where(mask[...,None], img, 0)
            # plt.imshow(finalImage)
            finalCombineImage_InfoDic[name] = [finalImage, mask]   

        if develop == True and i == 4:
            break
    # print(f'Total combined images: {len(finalCombineImage_InfoDic)}')
    return finalCombineImage_InfoDic

def imgMod(directory, develop,
           plotAllSpheroids, plotSingleSpheroid):

    """
    Takes in the directory of the image and mask files
    --> develop, plotAllSpheroids, plotSingleSpheroid are used for development purpose

    saves dictionary of the image as follows:
    - dictionary data
        -- combinedImage = data[0]
        -- mask = data[1]
        -- markedImage = data[2]
        -- spheroidInfoDic = data[3]   
            ---- spheroidInfoDic = {spheroidName: [area, centroids[i], croppedImage]}
            ---- area = spheroidInfoDic[spheroidName][0]
            ---- centroids = spheroidInfoDic[spheroidName][1]
            ---- croppedImage = spheroidInfoDic[spheroidName][2]


    Returns the array of marked finallyCombinedImage it's mask and path for annotation code
    also the name of the image in a dictionary format (key: name, value: finalImage, mask)

    """
    
    # get the list of the image and mask files
    targetMaskList = []
    inputImageList = []
    def imageList(directory):
        files = os.listdir(directory)
        imageList = []
        for file in files:
            imageList.append(file)
        return imageList
    
    if develop == True:
        print('this is develop mode for combined image info dictionary')
    
    files = os.listdir(directory)
    # print(files, len(files))
    for file in files:
        # print(f'image folder name{file}')
        if file == '.DS_Store':continue
        if file == 'Images_target':
            targetDir = os.path.join(directory, file)
            targetMaskList = imageList(targetDir)
        elif file == 'Images_input':
            inputDir = os.path.join(directory, file)
            inputImageList = imageList(inputDir)
        else:
            print("No such directory")

    finalCombineImage = {}
    finalCombineImage = combineInputMask(inputImageList, inputDir, 
                                         targetMaskList, targetDir, develop = develop)
    spheroidDetailPath = os.getcwd() + '/bigImg_spheroidDetails'

    if os.path.exists(spheroidDetailPath):
          print(f'Total {len(os.listdir(spheroidDetailPath))} image/mask dictionary present in\n {spheroidDetailPath}')    
    else:
         os.mkdir('bigImg_spheroidDetails')
         print(f'Created new folder {spheroidDetailPath} \nand should have {len(finalCombineImage)} files')

    divideImgPath = r'bigImg_spheroidDetails/'

    for key, value in finalCombineImage.items():
        img = value[0].copy()
        mask = np.array(value[1]>0, dtype=np.uint8)
        labelledSpheroids, spheroidInfoDic = divideAndLabelSpheroids(key, img, mask, 
														   plotAllSpheroids = plotAllSpheroids, 
                                                           plotSingleSpheroid = plotSingleSpheroid) 
        with open (divideImgPath + f'divided_{key}', 'wb') as f:
            pickle.dump((img, mask, labelledSpheroids, spheroidInfoDic), f)
            

    return finalCombineImage , spheroidDetailPath

 

def main():
    print("main function")
    args = parseArgs()
    print(args)
    imgMod(args.imageDirectory, args.develop,
           args.plotAllSpheroids, args.plotSingleSpheroid)


if __name__ == '__main__':
    main()

