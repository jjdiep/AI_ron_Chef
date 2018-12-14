import flickrapi
import urllib.request
from PIL import Image
import pdb
import cv2
import numpy as np






def find_similarity(img1,img2):
    #using sift brute-force
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #pdb.set_trace()

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)

    similar_regions = [i for i in matches if i.distance < 80]
    if len(matches) == 0:
      return 0
    return len(similar_regions) / len(matches)
    #return mean / (len(matches)*100)





# Flickr api access key 
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)


keyword = ['lemon']
#keyword = 'Italy cathedral'

photos = flickr.walk(text='lemon',
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     has_geo = 1,
                     per_page=1000,            # may be you can try different numbers..           
                     sort='relevance')

'''photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     extras=['url_c','location'],
                     per_page=100,            # may be you can try different numbers..           
                     sort='relevance')'''

urls = []
ids = []
locations = []
count = 0
for i, photo in enumerate(photos):
    print (i)
    #pdb.set_trace()
    url = photo.get('url_c')
    if url == None:
        continue
    else:
        id_c = photo.get('id')
        out_loc = flickr.photos_geo_getLocation(photo_id = id_c)
        if out_loc == None:
            continue
        else:
            ids.append(id_c)
            urls.append(url)
            loc_ = out_loc.find('photo').find('location')
            lat_ = loc_.attrib['latitude']
            lng_ = loc_.attrib['longitude']
            loc = [lat_,lng_]
            locations.append(loc)

            count += 1
            #pdb.set_trace()
    #info = flickr.photos.getInfo('id_c')
    #location = info.get('location')
    #locations.append(location)

    
    # get 500 urls
    if count > 50:
        break

print (urls)

#pdb.set_trace()
count = 0
# Download image from the url and save it to '00001.jpg'
for i in range(len(urls)):
    if urls[i] == None:
        continue
    else:
        count += 1
        urllib.request.urlretrieve(urls[i], str(count)+'.png')
        # Resize the image and overwrite it
        image = Image.open(str(count)+'.png')
        image = image.resize((1024, 1024), Image.ANTIALIAS)
        image.save('images/'+str(count)+'.png')
pdb.set_trace()

image = Image.open('data/image_of_cathedral.jpg')
image = image.resize((1024, 1024), Image.ANTIALIAS)
image.save('images/Test.png')


similarity = np.zeros((count,1))
#img_test = cv2.imread('images/Test.png', cv2.IMREAD_GRAYSCALE)
img_test = cv2.imread('images/Test.png')
for i in range(count):
    #img_train = cv2.imread('images/'+ str(i+1)+'.png', cv2.IMREAD_GRAYSCALE)
    img_train = cv2.imread('images/'+ str(i+1)+'.png')
    similarity[i] = find_similarity(img_test, img_train)

pdb.set_trace()
index = np.argsort(-similarity, axis = 0)
print(locations[index[0]])

