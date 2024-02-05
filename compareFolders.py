# python3 compareFolders.py -h
# python3 compareFolders.py -cdf -cF /Users/sabyasachi/Documents/new_change_1 -pF /Users/sabyasachi/Documents/new_change_2

import os 
import re
import sys
import argparse
from argparse import ArgumentParser, Namespace
import shutil



"""
    -- Data list 
        - This module will check all the files in two differeent directory and compare them
            - The targets and inputs will be moved to another directory 
            - In next iteration, it will move only new data and not old ones 
            - esch iteration will create a new directory with version number

"""
def parseArgs():
    # https://realpython.com/command-line-interfaces-python-argparse/
    parser = argparse.ArgumentParser(description="Takes in arguemnst for files transfer from one folder to another")
    parser.add_argument('-i', "--input", help="Input directory of the files", 
                        default = os.getcwd(), dest = 'inputFolder', type = str)
    parser.add_argument('-o', "--output", help="If you want to save the output in a different directory", action = 'store_true') 
    parser.add_argument('-cdf', '--compareDiffFolders', help = "Folders whose files will be compared, cFvspF", action = 'store_true')
    parser.add_argument('-cF', "--childFolder", required='-compareDiffFolders', 
                        help = "Child Folder that receives new data from Parent folder", type = str)
    parser.add_argument('-pF', "--parentFolder", required='-compareDiffFolders', 
                        help = "Parent Folder from which the data is transfered to child Folder", type = str)

    args = parser.parse_args()
    return args


def inputDirectory(inpDec):
    print('\nthis is the input directory: {0}'.format(inpDec))
    # get the input directory given by the user
    # inpDec = parseArgs.input

    files = os.listdir(inpDec)
    print(files, len(files))

    return files

def addnum(a,b):
    return int(a)+int(b)

def compareTransferFolderData(cFolder, pFolder = None, type = None):
    print('\nCompare transfer Function: transfers data from parent to child folder',
          '\nif new files are present in parent folder')

    if (type == 'diff'):
        print(f"Compare files of \n{cFolder}  \n{pFolder}")
        cfiles1 = os.listdir(cFolder)
        pfiles2 = os.listdir(pFolder)
        print(f'\ntotal files in',
              f'\n      -Child--->{cFolder} are {len(cfiles1)}',
              f'\n      -Parent--->{pFolder} are {len(pfiles2)}\n')
        
        commonFiles = list(set(cfiles1).intersection(set(pfiles2)))
        print(f'\nCommon files are {len(commonFiles)}')
        if len((cfiles1)) > len(pfiles2):
            print(f'\nChild folder has more files than parent folder')
        elif len((cfiles1)) < len(pfiles2):
                print(f'\nParent folder has more files than child folder')
                # (partent - child) folder
                differentFiles = list(set(pfiles2).difference(set(cfiles1)))
                print(f'      -Different files are {len(differentFiles)}')
                print(f'      ---> {differentFiles}\n\n')
                # move the different files to child folder
                for i in differentFiles:
                    print('ParentPath-->', os.path.join(pFolder, i), '\nChildPath-->', os.path.join(cFolder, i), '\n')
                    shutil.copy2(os.path.join(pFolder, i), os.path.join(cFolder, i))
        else :
            print(f'\nBoth folders have same number of files')
            print(f'\nNo files to move')
        commonFiles = list(set(cfiles1).intersection(set(pfiles2)))
        differentFiles = list(set(pfiles2).difference(set(cfiles1)))
    return commonFiles, differentFiles

def outputDirectory(fileName = None):
    if fileName == None :
        print('Name for the output directory is not given')
        return 
    else:
        cwd = os.getcwd()
        newDir = os.path.join(cwd, fileName)
        print('This is the output directory: {0}'.format(newDir))
        if not os.path.exists(newDir):
            os.makedirs(newDir)
            print('Created new directory')
        else :
            print('Directory already exists')
    return newDir



def main():
    print("This is the main function")
    args = parseArgs()
    # addnum(0,0)

    if args.compareDiffFolders == True:
        print("\nStart of with comparing the items in the different folders")
        compareTransferFolderData(args.childFolder, args.parentFolder, "diff")
    # outputDirectory("ERSRDFTGYJH")

    if args.output == True:
        print("Start of with the output directiry function")
        # newOutputFolder = "decide a name for the output folder"
        outputDirectory()

if __name__ == "__main__":
    main()
   