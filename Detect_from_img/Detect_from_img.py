

### Made By Hamrakulov Shohjahon. Car plate number detection from image ### 


import cv2
import numpy as np
import pytesseract

print("Running.....")

pytesseract.pytesseract.tesseract_cmd = 'D:/Programs/Programming/Pytesseract/tesseract.exe'  ### Loading tesseract character recognition

carplate_haar_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")  ### Leading Plate number detector

img = cv2.imread('Images/image1.jpg') ### Loading Images from Image folder

carplate_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ### make it gray




carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5) ### Detecting car number


if len(carplate_rects) > 0:  ### check if there is a car plate number in picture
    
    
    for x,y,w,h in carplate_rects: ### gets four points of car number plate
        
        carplate_img = carplate_img_rgb[y+15:y+h-10 ,x+15:x+w-20] ### cutting car plate
        

    carplate_extract_img = carplate_img

    scale_percent = 150 ### 150% to zoom

    ### Some calculations, Magic))

    width = int(carplate_extract_img.shape[1] * scale_percent / 100)
    height = int(carplate_extract_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    carplate_extract_img = cv2.resize(carplate_extract_img, dim, interpolation = cv2.INTER_AREA)



    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)

    carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray,3) # character size 3

    ### extracting characters from car plate

    text = pytesseract.image_to_string(carplate_extract_img_gray_blur, config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    text = text[:-2]

    ### put rectangle and car number
    
    for x,y,w,h in carplate_rects:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5)
        cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
        
    cv2.imshow("Image",img)

          
    print("Detected Number:", text)

else:
    
    print("Car licence number is not found!")


