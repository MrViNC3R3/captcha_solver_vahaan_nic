from keras.models import load_model
#from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from PIL import Image 
import matplotlib.pyplot as plt

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

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
total = len(captcha_image_files)
# loop over the image paths
for image_file in captcha_image_files:
    img = Image.open(image_file)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(image,cmap='gray')
    #plt.show()
    #break    
    filename=image_file
    filename = filename.replace('.jpg','')
    filename = filename.replace('capt/','')
    print(filename)
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
        continue

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

    # Show the annotated image
    #cv2.imshow("Output", output)
    #cv2.waitKey()
    if captcha_text == filename:
        correct = correct+1
    else:
        wrong = wrong+1
print('Total  = '+str(total))
print('Correct  = '+str(correct))
print('Wrong    = '+str(wrong))