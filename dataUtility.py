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
    parser.add_argument('-dil', "--_dilation", help="If you want to dilate the mask for the extra curves", default = False)
    parser.add_argument('-dev', "--develop", help="Develpoment mode or not", action = 'store_true')  

    args = parser.parse_args()
    return args



# def divideImage(image):
#     return imageFolder

def fullLabelledImage(segImg, numLabels, stats, centroids, plotAllSpheroids = False):
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
			cv2.rectangle(segImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(segImg, str(i), (int(cX), int(cY)), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
			# cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), i) #add the centroid to the image
	if plotAllSpheroids == True:		
		plt.figure(figsize=(10,10))
		plt.imshow(segImg)
		plt.title(f"Total Embryonic Bodies = {numLabels-1}", fontsize = 10)
		plt.show()
	return segImg

	
def singleSpheroidImage(name, mask, oriImage, numLabels, 
                        labels, stats, centroids, plotSingleSpheroid = False, _dilation = False, develop = False):
    
# to do : dilate the individual embroyonic bodies more by 5 iterations and then add them to the dictionary
    # : add the individual mask to the dictionary but wihout the dilating  
    # so during training we can use the mask for the segmentation once more 
    spheroidInfoDic = {}
    image = np.where(mask[...,None], oriImage, 0)
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
          if i == 0:
                text = "examining component {}/{} (background)".format(i , numLabels-1)
                # print("[INFO] {}".format(text))
                continue
          elif develop == True and i==6: break
          else:
                text = "examining component {}/{}".format( i, numLabels-1)
                # print("[INFO] {}".format(text))
                
                spheroidName = str(f'spheroid{i}_' + name)
                # extract the stats for oriMask
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]
                # create the new stats for expanded cropped image
                x_ = max((x - 10), 0)
                y_ = max((y - 10), 0)
                w_ = min((w + 20), oriImage.shape[1] - x_)
                h_ = min((h + 20), oriImage.shape[0] - y_)

                # Extracting Embryonic cluster pre segmentaton - with the original background
                cropped_oriImage = np.array(oriImage[y_:y_+h_, x_:x_+w_, :])
                # croppedMask = np.array(mask[y:y+h_, x:x+w_])

                # extracting the Embryonic cluster post segmentation - with the black background
                cropped_segImage = np.array(image[y:y+h, x:x+w, :])
                # make it part of the argument or based on segmentaton result
                if _dilation == True:
                     componentMask = (labels == i).astype("uint8") * 255
                     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                     componentMask = cv2.dilate(componentMask, kernel, iterations=3)
                     cropped_segMask = np.where(componentMask[y:y+h, x:x+w], (np.array(mask[y:y+h, x:x+w])), 0)
                     cropped_segImage = np.where(componentMask[y:y+h, x:x+w, None], cropped_segImage, 0)
                     # for the dilated image different cropping measuremnts are used
                     dilated_img = np.where(componentMask[y_:y_+h_, x_:x_+w_, None], np.array(oriImage[y_:y_+h_, x_:x_+w_, :]), 0)
                    #  spheroidInfoDic[spheroidName] = [area, centroids[i], cropped_oriImage, dilated_img ,cropped_segMask, componentMask]
                else:
                     componentMask = (labels == i).astype("uint8") * 255
                     cropped_segMask = np.where(componentMask[y:y+h, x:x+w], np.array(mask[y:y+h, x:x+w]), 0)
                     cropped_segImage = np.where(componentMask[y:y+h, x:x+w, None], cropped_segImage, 0)
                    #  spheroidInfoDic[spheroidName] = [area, centroids[i], cropped_oriImage, cropped_segImage ,cropped_segMask, componentMask]
                    
                spheroidInfoDic[spheroidName] = [(area,x,y,h,w,centroids[i]), cropped_oriImage, 
                                                 cropped_segMask, componentMask, cropped_segImage] 

                if plotSingleSpheroid == True:
                        # plt.figure(figsize=(5,5))
                        fig, ax = plt.subplots(1, 4, figsize=(8,4))
                        ax[0].imshow(cropped_oriImage)
                        ax[0].set_title('Cropped Original Image', fontsize=5)

                        ax[1].imshow(cropped_segImage)
                        ax[1].set_title('Cropped Segmented Image', fontsize=5)

                        ax[2].imshow(dilated_img)
                        ax[2].set_title('Dilated mask Image', fontsize=5)

                        ax[3].imshow(componentMask)
                        ax[3].set_title(f"Area = {area}; \nSpheroid Number = {i}", fontsize = 5)
                        ax[3].axis('off')

                        plt.tight_layout()
                        plt.show()
    
    return spheroidInfoDic
     

def divideAndLabelSpheroids(name, mask, oriImage, 
                            plotAllSpheroids = False, plotSingleSpheroid = False, _dilation = False, develop = False):
      """
      Takes in the image and mask and returns the image with the labelled spheroids
      """
														   
      output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
      (numLabels, labels, stats, centroids) = output
      
      segImg = np.where(mask[...,None], oriImage, 0)
      ## display this one in a separate cell for user to get the whole picture and make plotAllSpheroids = True
      labelledSpheroids = fullLabelledImage(segImg, numLabels, stats, centroids, plotAllSpheroids)

      spheroidInfoDic = singleSpheroidImage(name, mask, oriImage,
									numLabels, labels, stats, centroids, plotSingleSpheroid, _dilation, develop)
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
        oriImg = np.array(Image.open(os.path.join(inputDir, name)))
        mask = '.'.join(name.split('.')[:-1]) + "_bn.tif"
        if mask in targetMaskList:
            mask = np.array(Image.open(os.path.join(targetDir, mask)))
            if develop == True and i == 5:
                # if _dilation == True: # perform dilation on the complete mask
                #      mask = cv2.dilate(mask, kernel, iterations=5)
                #      finalCombimeImage_InfoDic[name] = [mask, oriImg]
                finalCombineImage_InfoDic[name] = [mask, oriImg]
                break
            else: 
                finalCombineImage_InfoDic[name] = [mask, oriImg]
        else: 
            print(f'No mask found for {name}')
            continue 
        
    return finalCombineImage_InfoDic

def imgMod(directory, develop,
           plotAllSpheroids, plotSingleSpheroid, _dilation):

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
             print(f"{file} not found - {directory}")

    combineImageInfo = {}
    spheroidDetailPath = os.getcwd() + '/bigImg_spheroidDetails'
    combineImageInfo['path'] = spheroidDetailPath
    combineImageInfo['ImageList'] = combineInputMask(inputImageList, inputDir, 
                                                      targetMaskList, targetDir, develop = develop)
    
    if os.path.exists(spheroidDetailPath):
          print(f'Total {len(os.listdir(spheroidDetailPath))} images and their details present in\n {spheroidDetailPath}')    
    else:
         os.mkdir('bigImg_spheroidDetails')
         print(f"Created new folder {spheroidDetailPath} \nand should have {len(combineImageInfo['ImageList'])} files")

    divideImgPath = r'bigImg_spheroidDetails/'

    for name, value in combineImageInfo['ImageList'].items():
        mask = np.array(value[0], dtype=np.uint8)
        oriImage = value[1].copy()
        labelledSpheroids, spheroidInfoDic = divideAndLabelSpheroids(name, mask, oriImage,
                                                                     plotAllSpheroids = plotAllSpheroids, 
                                                                     plotSingleSpheroid = plotSingleSpheroid,
                                                                     _dilation = _dilation,
                                                                     develop = develop)
        with open (divideImgPath + f'divided_{name}', 'wb') as f:
             pickle.dump((mask, oriImage, spheroidInfoDic, labelledSpheroids), f)
  

    return combineImageInfo

 
# def csvToDictionary(annotatedSheroidPath, spheroidName):
#     """
#     This function takes in the annotatedSheroidPath and spheroidName
#     returns the dictionary of the csv file
#     """
#     csvPath = os.path.join(annotatedSheroidPath, spheroidName, 'labels.csv')
#     df = pd.read_csv(csvPath)
#     df = df.drop(['Unnamed: 0'], axis = 1)
#     df = df.rename(columns = {'image':'image_name'})
#     df = df.set_index('image_name')
#     df = df.to_dict('index')
#     return df


def main():
    print("main function")
    args = parseArgs()
    print(args)
    imgMod(args.imageDirectory, args.develop,
           args.plotAllSpheroids, args.plotSingleSpheroid, args._dilation)


if __name__ == '__main__':
    main()

