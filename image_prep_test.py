#code part 1
from bs4 import BeautifulSoup  # pip install bs4
import numpy as np
import requests
import cv2
import PIL.Image  # pip install Pillow
import urllib
import pdb
import os

'''
Credit to ImageNet project for compiling images and feature information
Note: did not include non-bbox ImageNet subsets, just there for reference
snap bean- no bbox
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07728053
potato- no bbox
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07710616
carrot - no bbox
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07730207
ginger - no bbox
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n12356023
eggplant - no bbox
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07713074
corn - n07732168
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07732168
cherry tomato - n07734017
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07734017
egg - n07840804
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07840804
green onion - n07722485
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07722485
shallot - n07723177
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07723177
spinach - n07736692
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07736692
spinach beet - n07720277 - this one is very inaccurate and was not used
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07720277
cheese - n07850329
http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07850329
shiitake mushroom - n13001930
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n13001930
broccoli - n07714990
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07714990
garlic - n07818277
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07818277
cucumber - n07718472
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07718472
bell pepper - n07720875
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07720875
leek - n07723039
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07723039
bok choy - n07714448
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07714448
cauliflower - n07715103
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07715103
hummus - n07857731  # not really an ingredient, taken out
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07857731
lettuce - n07723559
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07723559
Head cabbage - n07714571
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07714571
apple - n07739125
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07739125
lemon - n07749582
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07749582

Generic BBOX address - http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=[WNID]
Generic Image URL address - http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[WNID]
'''

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

def url_extractor(wnid):
    # wnid = "n04194289"  # ship
    # page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07730207")
    page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+wnid)
    # page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+synid)  # ship synset (change to food ingredient)

    # print(page.content)

    # BeautifulSoup is an HTML parsing library

    soup = BeautifulSoup(page.content, 'html.parser')  # puts the content of the website into the soup variable, each url on a different line
    #print(soup)
    # print(soup.prettify())  # check if url is valid

    #code part 2
    str_soup=str(soup)  # convert soup to string so it can be split
    type(str_soup)
    split_urls = str_soup.split('\r\n')#split so each url is a different position on a list
    print('Number of Total URLs:', len(split_urls))#print the length of the list so you know how many urls you have

    return split_urls

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if np.mean(image) > 253:  # certain html links from imagenet no longer work
        image = 0
        print('Image no longer exists') 
    # return the image
    return image

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

def img_gen(directory, n_of_training_images, img_split_urls):
    print('Creating Images in', directory)
    real_progress = 0
    progress = 0
    valid1 = 'http://farm'
    valid2 = 'http://static'
    pdb.set_trace()
    while real_progress < n_of_training_images:#store all the images on a directory
        # Print out progress whenever progress is a multiple of 20 so we can follow the
        # (relatively slow) progress
        if(progress%2==0):
            print(progress)
        if not img_split_urls[progress] == None:
            try:
                # take only Flickr images (for some quality control...)
                if img_split_urls[progress][:len(valid1)] == valid1 or img_split_urls[progress][:len(valid2)] == valid2:
                    I = url_to_image(img_split_urls[progress])

                    if (len(I.shape)) == 3: #check if the image has width, length and channels
                        save_path = directory+'img_'+str(real_progress)+'.jpg'#create a name of each image
                        cv2.imwrite(save_path,I)
                        real_progress += 1
            except:
                None
        progress += 1

# def bbox_finder(wnid):
#     path = wnid

def main():

    n_img = 100 # choose the number of training images to use, make sure not to pick more images than there are urls!
    word, wnid = word_finder()

    for num_words in range(len(word)):
        print(str(num_words)+' out of '+str(len(word)))
        pdb.set_trace()
        directory = dir_maker(word[num_words])
        img_split_urls = url_extractor(wnid[num_words])
        img_gen(directory, n_img, img_split_urls)

    # demo code, should run on apple class

    # directory = dir_maker(word[0])
    # img_split_urls = url_extractor(wnid[0])
    # img_gen(directory, n_img)

    print('Done!')



if(__name__=="__main__"):
    main()
