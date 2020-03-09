import requests
from PIL import Image
import os
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import time
from keras.models import load_model
#from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import csv


# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
#captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
#total = len(captcha_image_files)
# loop over the image paths
#for image_file in captcha_image_files:
def cap_sol(image_file):
    img = Image.open(image_file)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(image,cmap='gray')
    #plt.show()
    #break
    #filename=image_file
    #filename = filename.replace('.jpg','')
    #filename = filename.replace('capt/','')
    #print(filename)
    #print(len(filename))
    ii = []
    left = 0
    i=0
    #print(main_counter)
    #if len(filename) <6 or len(filename) >6:
    #    continue
    right=0
    width,height = img.size
    img = img.crop((8,0,width-12,height))
    print(img.size)
    width,height = img.size
    #io.show()
    u=0
    letter_image_regions = []
    while i <=68.33:
        io = img.crop((left+i,right,13.66+i,height))
        #io.show()
        i=i+13.66
        io = io.resize((150,150))
        io = io.point(lambda x: 0 if x<195 else 255, '1')
        io.save("extr/"+str(u)+'.png')
        #letter_image_regions.append(io)
        u=u+1
    letter_image_regions = []
    #image = cv2.imread(image_file)
    #main_counter = main_counter+1
    """# Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))
    """

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    extr  = list(paths.list_images('extr'))
    #print(extr)
    extr.sort()
    #print(extr)
    for iii in extr:
        iiu = cv2.imread(iii)
        iiu = cv2.cvtColor(iiu, cv2.COLOR_BGR2GRAY)
        letter_image_regions.append(iiu)

    if len(letter_image_regions) != 6:
        print('6 letter error')
        return('error')

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    #letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    #output = cv2.merge([image] * 3)
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        #x, y, w, h = letter_bounding_box
        letter_image = letter_bounding_box
        # Extract the letter from the original image with a 2-pixel margin around the edge
        #letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = cv2.resize(letter_image, (110, 110))

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)
        #print(predictions)

        # draw the prediction on the output image
        #cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        #cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
    return(captcha_text)







MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "capt"
correct = 0
wrong = 0

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)



chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(executable_path='chromedriver',chrome_options=chrome_options)
driver.set_window_size(1920, 1376)

"""driver.get('https://vahan.nic.in/nrservices/faces/user/searchstatus.xhtml')
el = driver.find_element_by_class_name('captcha-image')
location = el.location
print(str(location))
size = el.size
print(str(size))
driver.save_screenshot('cap_img.jpg')

x = location['x']
y = location['y']
w = size['width']+2
h = size['height']
width = x + w
height = y + h
im = Image.open('cap_img.jpg').convert("L")
im = im.crop((int(x), int(y), int(width), int(height)))
im.save('cap_img.jpg')
captcha_text = cap_sol('cap_img.jpg')"""
#key = 'pzhvjrdfm6nwp4ygyngjtbrtzcfx2wvk'
states = ['ka','ap','ar','br','cg','ch','an','dl','dn','dd','ga','gj','hp','hr','jh','jk','kl','ld','la','mh','ml','mn','mp','mz','nl','od','pb','py','rj','sk','tn','ts','tr','up','uk','wb']

while 1:
        for state in states:
                for rto in range(5,80):
                        heade = []
                        info = {'Registration No':'','Registration Date':'','Chassis No':'','Engine No':'','Owner Name':'','Vehicle Class':'','Fuel':'','Maker / Model':'','Fitness/REGN Upto':'','MV Tax upto':'','Insurance Details':'','PUC No / upto':'','Emission norms':'','RC Status':''}
                        for key,value in info.items():
                                heade.append(key)
                        print(heade)
                        with open('data/'+str(state)+'_'+str(rto)+'.csv','a') as csv_f:
                                writer = csv.writer(csv_f)
                                writer.writerow(heade)
                        for alp in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
                                for alp1 in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
                                        for z in range(10):
                                                for t in range(10):
                                                        for ii in range(10):
                                                                for u in range(1,10):
                                                                        while 1:
                                                                                try:
                                                                                        driver.get('https://vahan.nic.in/nrservices/faces/user/searchstatus.xhtml')
                                                                                        el = driver.find_element_by_class_name('captcha-image')
                                                                                        location = el.location
                                                                                        print(str(location))
                                                                                        size = el.size
                                                                                        print(str(size))
                                                                                        driver.save_screenshot('cap_img.jpg')
                                                                                        print(u)
                                                                                        x = location['x']
                                                                                        y = location['y']
                                                                                        w = size['width']+2
                                                                                        h = size['height']
                                                                                        width = x + w
                                                                                        height = y + h
                                                                                        im = Image.open('cap_img.jpg').convert("L")
                                                                                        im = im.crop((int(x), int(y), int(width), int(height)))
                                                                                        im.save('cap_img.jpg')
                                                                                        captcha_text = cap_sol('cap_img.jpg')
                                                                                        if int(rto) < 10:
                                                                                                rto_1 = '0'+str(rto)
                                                                                        else:
                                                                                                rto_1 = str(rto)
                                                                                        vehicle_no = str(state)+str(rto_1)+str(alp)+str(alp1)+str(z)+str(t)+str(ii)+str(u)
                                                                                        print(vehicle_no)
                                                                                        driver.find_element_by_id('regn_no1_exact').send_keys(str(vehicle_no))
                                                                                        driver.find_element_by_id('txt_ALPHA_NUMERIC').send_keys(str(captcha_text))
                                                                                        driver.find_element_by_class_name('ui-button-text').click()
                                                                                        time.sleep(4)
                                                                                        i=0
                                                                                        driver.save_screenshot('initial.png')
                                                                                        data = driver.find_element_by_tag_name('tbody')
                                                                                        details = []
                                                                                        flag = 0
                                                                                        #info = {'Registration No':'','Registration Date':'','Chassis No':'','Engine No':'','Owner Name':'','Vehicle Class':'','Fuel':'','Maker / Model':'','Fitness/REGN Upto':'','MV Tax upto':'','Insurance Details':'','PUC No / upto':'','Emission norms':'','RC Status':''}
                                                                                        for rows in data.find_elements_by_tag_name('tr'):
                                                                                                for iu in  rows.find_elements_by_tag_name('td'):
                                                                                                        #print(i)
                                                                                                        if i==0:
                                                                                                                print(iu.text)
                                                                                                                i = i+1
                                                                                                                continue
                                                                                                        if i % 2 != 0 :
                                                                                                                dd = (iu.text).replace(":","")
                                                                                                        else:
                                                                                                                print(iu.text)
                                                                                                                details.append(iu.text)
                                                                                                                if "RC Status" in dd:
                                                                                                                        flag = 1
                                                                                                                        break
                                                                                                        i = i+1
                                                                                                if flag == 1:
                                                                                                        break
                                                                                        print(details)
                                                                                        with open('data/'+str(state)+'_'+str(rto)+'.csv','a+') as csv_file:
                                                                                                writer = csv.writer(csv_file)
                                                                                                writer.writerow(details)
                                                                                                #for key, value in info.items():
                                                                                                #       writer.writerow([key, value])
                                                                                        break
                                                                                except Exception as e:
                                                                                        try:
                                                                                                error_type = driver.find_element_by_class_name('ui-messages-info-summary').text
                                                                                                print(error_type)
                                                                                                if 'Verification Code' in error_type:
                                                                                                        print('Again')
                                                                                                        continue
                                                                                                if 'Vehicle Detail not found' in error_type:
                                                                                                        print('invalid')
                                                                                                        break
                                                                                        except:
                                                                                                pass
