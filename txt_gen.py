import numpy as np
import os

# tmp script for generating train.txt and classes.txt

def word_finder():
    # needs wnid.txt in same directory, wnid holds all desired imagenet id words and addresses
    fh = open("wnid.txt", "r")

    # To read one line at a time, use:
    # print fh.readline()

    # To read a list of lines use:
    wnid_list = fh.readlines()
    
    print(wnid_list)
    word = []
    wnid = []
    for line in range(len(wnid_list)):
        item = wnid_list[line].split(' ')
        word.append(item[0])
        wnid.append(item[1][:-1])

    fh.close()
    return word, wnid

word, wnid = word_finder()
fh1 = open("classes.txt", "w")
for line1 in range(len(word)):
    fh1.write(word[line1]+'\n')

fh1.close()

fh2 = open("train.txt", "w")
for line2 in range(len(word)):
    for img_num in range(50): # needs to change with number of training images
        fh2.write(word[line2]+'/img_'+str(img_num)+'\n')
fh2.close()

'''
def dir_maker(word):
    # evidently Linux and Windows dependent but...
    path = 'data/train/' + word +'/'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, path)

    if not os.path.exists(final_directory):
        try:
            os.makedirs(final_directory)
        except OSError:
            print("Creation of the directory failed" % path)
        else:
            print("Successfully created the directory %s" % path)

    return final_directory
'''
