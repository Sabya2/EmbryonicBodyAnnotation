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
import numpy as np
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
                                options = labels, include_next= False,
                                buttons_in_a_row = 4, shuffle=False,
                                final_process_fn = finalProcessing, 
                                display_fn = lambda filename: custom_display(filename))
    return annotations

# add a same button for all the images
labels = ['1a-EmbryonicBody', '1b-NonEmbryonicBody', '2a-LocationEdge', '2b-LocationCenter',
          '3a-OverlappingEB', '3b-NonOverlappingEB', '4a-ConnectedEB','4b-DisconnectedEB',
          '5a-SmoothOutline', '5b-RoughOutline', '6a-DenseEB', '6b-LightEB', 
          '7a-NonCytstic', '7b-Cytstic', '7c-HeavilyCystic', 'Add_NewProperty',
          '8a-IrregularShaped', '8b-OvalShaped', '8c-RoundShaped', 
          "Same_As_previous", 'Cropped_Detail', "Can't_determine"]


def annotastionCSV(imgPath, csvPath, spheroidInfo):
    annotations = MultiClass_spheroidLabelling(imgPath, csvPath,
                                            labels = labels, 
                                            annoteType = 'multilabel-classification',
                                            spheroidInfo = spheroidInfo)



def savingImages(mask, data, path, develop):

    """
    this Fucntion takes in the spheroidInfo dictionary 
    saves the smaller embryonic bodies and returns their path in a list

    -- data = spheroidInfoDic  
            ---- spheroidInfoDic = {spheroidName: [(area,x,y,h,w,centroid), cropped_oriImage, cropped_segImage, cropped_segMask, componentMask]}
            ------# area,x,y,h,w,centroid = spheroidInfoDic[spheroidName][0]
            ------# cropped_oriImage = spheroidInfoDic[spheroidName][1]
            ------# cropped_segMask = spheroidInfoDic[spheroidName][2]
            ------# componentMask = spheroidInfoDic[spheroidName][3]
            ------# cropped_segImage = spheroidInfoDic[spheroidName][4]

    """

    imgPathList = []
    i = 0
    for key, value in data.items():
        i = i+1
        cropped_oriImage  = cv2.cvtColor(value[1], cv2.COLOR_RGB2BGR)
        cropped_segImage  = cv2.cvtColor(value[4], cv2.COLOR_RGB2BGR)
        cropped_segImage = cv2.resize(cropped_segImage, (cropped_oriImage.shape[1], cropped_oriImage.shape[0])) 
        componentMask = value[3]

        imageHeight = cropped_oriImage.shape[0]
        imageWidth = cropped_oriImage.shape[1]
        combined_image = np.zeros((imageHeight*2 , imageWidth * 4, 3), dtype = np.uint8)


        componentMask = cv2.resize(componentMask, (int(combined_image.shape[1]/3), int(combined_image.shape[0]/1.5)))
        componentMask = np.expand_dims(componentMask, axis=2)

        wGap = int(combined_image.shape[1]/20)
        hGap = int(combined_image.shape[0]/10)

        borderHeight = 100
        combined_image = cv2.copyMakeBorder(combined_image, borderHeight,0,0,int(combined_image.shape[1]/2), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        text = f"{key.split('_')[0]} - (Area:{value[0][0]})"
        combined_image = cv2.putText(combined_image, text, (0, 50), 
                                     cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0,0,0), 1, cv2.LINE_AA) 


        text = 'Original'
        combined_image[hGap*4+borderHeight:imageHeight+hGap*4+borderHeight, int(wGap/2):imageWidth+int(wGap/2)] = cropped_oriImage
        lineStart = (int(wGap/2)+imageWidth+15, borderHeight)
        lineEnd = (int(wGap/2)+imageWidth+15, imageWidth+combined_image.shape[1])
        cv2.line(combined_image, lineStart, lineEnd, (255, 255, 255), thickness=2)
        cv2.putText(combined_image, text, (wGap, 90), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.8, (0,0,0), 1, cv2.LINE_AA) 
        
        text = 'from_Model'
        combined_image[hGap*4+borderHeight:imageHeight+hGap*4+borderHeight, wGap*2+imageWidth:wGap*2+imageWidth*2] = cropped_segImage
        lineStart = (wGap*2+imageWidth*2 +15, borderHeight)
        lineEnd = (wGap*2+imageWidth*2 +15,imageWidth+combined_image.shape[1])
        cv2.line(combined_image, lineStart, lineEnd, (255, 255, 255), thickness=2)
        cv2.putText(combined_image, text, (wGap+imageWidth,90 ), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.8, (0,0,0), 1, cv2.LINE_AA) 
        
        text = 'Approx_location'
        cv2.putText(combined_image, text, (wGap*3+imageWidth*2,90), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.8, (0,0,0), 1, cv2.LINE_AA) 
        combined_image[hGap*2+borderHeight:componentMask.shape[0]+hGap*2+borderHeight, wGap*3+imageWidth*2:wGap*3+imageWidth*2+componentMask.shape[1]] = componentMask
    
        filename = path + '/' + key + '.png'
        cv2.imwrite(filename, combined_image)
        imgPathList.append(filename)

        if develop == True and i == 10:
            break

    return imgPathList


def foldersWithCSV(annotatedSheroidPath, dividedSpheroidDicPath):
    """ 
    This function takes in the annotatedSheroidPath 
    returns the list of folders with csv files
    and deletes the extra data colleted for annoataion 
            - namely the component mask, ropped_segImg, marled image
    """

    files = os.listdir(annotatedSheroidPath)
    annotatedList = []

    for file in files:
        folderPath = annotatedSheroidPath + '/' + file
        if (glob.glob(folderPath + '/*.csv') != []):
            ## here we can delete the extra data collected for annotation
            annotatedList.append(file)

    print(f'{len(annotatedList)} image/s completely annotated')
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
        annotatedList = foldersWithCSV(annotatedSheroidPath = annotatedSheroidPath,
                                        dividedSpheroidDicPath = dividedSpheroidDicPath )
    else : 
        os.mkdir(annotatedSheroidPath)
        annotatedList = []

    # random file selection for annotation based on the annotatedList
    if len(annotatedList) == 0:
        file = random.choice(files)
    elif len(annotatedList) == len(files):
        print('All the images are annotated')
        sys.exit() 
    else:
        randomList = [x for x in files if x not in annotatedList]
        file = random.choice(randomList)

    # open the random file and show the image 
    with open(dividedSpheroidDicPath + '/' + file, 'rb') as f:
        data = pickle.load(f)
        mask = data[0]
        # oriImage = data[1]
        spheroidInfoDic = data[2]
        markedImage = data[3]
        # stats(area,x,y,h,w,centroids[i]) = spheroidInfoDic[spheroidName][0]
        # cropped_oriImage = spheroidInfoDic[spheroidName][1]
        # cropped_segMask = spheroidInfoDic[spheroidName][2]
        # componentMask = spheroidInfoDic[spheroidName][3]
        # cropped_segImage = spheroidInfoDic[spheroidName][4]
    # fig,ax = plt.figure(figsize=(8,8))
    plt.rcParams["figure.figsize"] = [8,10]
    plt.imshow(markedImage)
    plt.title(f'{file} \nwith {len(spheroidInfoDic.keys())} potential emryonic bodies', y = -0.18)
    plt.show()

    filePath = annotatedSheroidPath + '/' + file
    if not os.path.exists(filePath):
        os.mkdir(filePath)

    if develop == True:
        print(f'Develop mode for the above image {file}\n New folder created to save the embryonic body images and annotated csv file') 

    imgPathList = savingImages(mask, spheroidInfoDic, filePath, develop = develop)
    # print('imagePath-',imgPathList) #images saved for annotation'
    annotastionCSV(imgPath = imgPathList, csvPath = filePath, spheroidInfo = str(file))


def main():
    print('main function')
    args = parseArgs()
    spheroidAnnotate(args.dividedSpheroidDicPath, args.develop)


if __name__ == "__main__":
    main()
    pass