# pyhton3 pigeonUtility.py -dSDP /Users/sabyasachi/Documents/internship_data/code/bigImg_spheroidDetails
# if i == 4: break
## search the above and remove it from the code



import pigeonXT as pixt
from IPython.display import display, Image
import os
import re
import glob
import random 
import argparse
import pickle 
import cv2
import pandas as pd
import shutil, sys 
import matplotlib.pyplot as plt

def parseArgs():
    parser = argparse.ArgumentParser(description="Spheroid annotations")
    parser.add_argument('-dSDP', "--dividedSpheroidDicPath", help="Input directory of the image and target files")
    parser.add_argument('-dev', "--develop", help="Develpoment mode or not", action = 'store_true')  
    args = parser.parse_args()
    return args

## Multilabel Classification simple
def MultiClass_spheroidLabelling(imgPath, csvPath, labels, annoteType, spheroidInfo):
    def custom_display(html_content):
         # Convert HTML object to string
         html_string = str(html_content)
         match = re.search(r'<div>(.*?)</div>', html_string)
         if match:
             image_path = match.group(1)
             display(Image(filename=image_path))
         else:
             print("No image path found in HTML content")

    def finalProcessing(annotations):
         filename = csvPath + '/' + f'labelled_{spheroidInfo}.csv'
         annotations.to_csv(filename, index=False)
         annotations = pd.read_csv(filename)
         annotations['example'] = annotations['example'].apply(lambda x: x.split('/')[-1])

         annotations.to_csv(filename, index=False)
         print(f'CSV file created for spheroid_{spheroidInfo}')

    annotations = pixt.annotate(imgPath, task_type = annoteType,
                                options = labels,
                                buttons_in_a_row = 4, shuffle=True,
                                final_process_fn = finalProcessing, 
                                display_fn = lambda filename: custom_display(filename))
    return annotations


labels = ['1-(EB) Embryonic Body', '1-(NEB) non-Embryonic Body',
          '2-Location-Edge', '2-Location-Center',
          '3-Overlapping-Yes', '3-Overlapping-No',
          '4-Outline-Smooth', '4-Outline-Rough',
          '5-Non-cystic', '5-Cytstic','5-Heavily-cystic',
          '6-Shape-Round', '6-Shape-Oval', '6-Shape-Irregular', "7-Can't determine"]

def annotastionCSV(imgPath, csvPath, spheroidInfo):
    annotations = MultiClass_spheroidLabelling(imgPath, csvPath,
                                            labels = labels, 
                                            annoteType = 'multilabel-classification',
                                            spheroidInfo = spheroidInfo)



def savingImages(data, path, develop):

    """
    this Fucntion takes in the spheroidInfo dictionary 
    saves the smaller embryonic bodies and returns their path in a list

    -- data = spheroidInfoDic  
            ---- spheroidInfoDic = {spheroidName: [area, centroids[i], croppedImage]}
            ---- area = spheroidInfoDic[spheroidName][0]
            ---- centroids[i] = spheroidInfoDic[spheroidName][1]
            ---- croppedImage = spheroidInfoDic[spheroidName][2]
    """

    imgPathList = []
    i = 0
    for key, value in data.items():
        i = i+1
        text = f"{key.split('_')[0]}-(Area:{value[0]})"
        org = (2, 25)
        croppedImage  = cv2.cvtColor(value[2], cv2.COLOR_RGB2BGR)
        bimg = cv2.copyMakeBorder(croppedImage, 40,0,0,300, cv2.BORDER_CONSTANT, value=[255, 255, 255]) 
        bimg = cv2.putText(bimg, text, org, cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.7, (0,0,0), 1, cv2.LINE_AA) 
        filename = path + '/' + key + '.png'
        cv2.imwrite(filename, bimg)
        imgPathList.append(filename)

        if develop == True and i == 4:
            break

    return imgPathList


def foldersWithCSV(annotatedSheroidPath):
    """ 
    This function takes in the annotatedSheroidPath 
    returns the list of folders with csv files
    """

    files = os.listdir(annotatedSheroidPath)
    annotatedList = []

    for file in files:
        folderPath = annotatedSheroidPath + '/' + file
        if (glob.glob(folderPath + '/*.csv') != []):
            # print(f'folders with csv present in {folderPath}')
            annotatedList.append(file)

    print(f'{len(annotatedList)} image/s completely annotated')
    # print(f'annotated images are {annotatedList}')
    return annotatedList


def spheroidAnnotate(dividedSpheroidDicPath, develop):
    """
    This function takes in the dividedSpheroidDicPath and returns the annotated images with csv file
    Creates a new folder (filePath---> annotatedSheroidPath/spheroid_Name/) for each image and svaes the mebryonic bodies for annotation
    """
    files = os.listdir(dividedSpheroidDicPath) # refers to bigImg_spheroidDetails folder
    annotatedSheroidPath = os.getcwd() + '/annotated_spheroids' # refers to annotated_spheroids folder

    # creation of the annotated_spheroids folder and list of annotated images
    if os.path.exists(annotatedSheroidPath):
        annotatedList = foldersWithCSV(annotatedSheroidPath = annotatedSheroidPath)
    else : 
        os.mkdir(annotatedSheroidPath)
        annotatedList = []

    # random file selection for annotation based on the annotatedList
    if len(annotatedList) == 0:
        file = random.choice(files)
    else:
        randomList = [x for x in files if x not in annotatedList]
        file = random.choice(randomList)

    # state of the code 
    if develop == False:
        print(f'Annotation mode for {file}')
    else:
        print(f'Develop mode for {file}')

    # open the random file and show the image 
    with open(dividedSpheroidDicPath + '/' + file, 'rb') as f:
        data = pickle.load(f)
        # combinedImage = data[0]
        # mask = data[1]
        markedImage = data[2]
        spheroidInfoDic = data[3]
        # area = spheroidInfoDic[spheroidName][0]
        # centroids = spheroidInfoDic[spheroidName][1]
        # croppedImage = spheroidInfoDic[spheroidName][2]
    fig = plt.figure(figsize=(8,8))
    plt.imshow(markedImage)
    plt.title(f'{file} with \n{len(spheroidInfoDic.keys())} potential emryonic bodies')
    plt.show()

    filePath = annotatedSheroidPath + '/' + file
    if not os.path.exists(filePath):
        os.mkdir(filePath)
    print(f'New folder created for {file} \nto save the embryonic body images and annotated csv file') 

    imgPathList = savingImages(spheroidInfoDic, filePath, develop = develop)
    annotastionCSV(imgPath = imgPathList, csvPath = filePath, spheroidInfo = str(file))


def main():
    print('main function')
    args = parseArgs()
    spheroidAnnotate(args.dividedSpheroidDicPath, args.develop)


if __name__ == "__main__":
    main()
    pass