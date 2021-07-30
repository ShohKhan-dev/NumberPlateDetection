

### Made By Hamrakulov Shohjahon. Car plate number detection from image ###


import cv2
import numpy as np
import pytesseract



pytesseract.pytesseract.tesseract_cmd = 'D:/Programs/Programming/Pytesseract/tesseract.exe' ### Loading tesseract character recognition

carplate_haar_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml") ### Leading Plate number detector


cap =cv2.VideoCapture(1) ### Loading video

print("Running....")

count = 0

while True:
    success , img  = cap.read()
    
    carplate_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ### make it gray


    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5)  ### Detecting car number


    if len(carplate_rects) > 0:  ### check if there is a car plate number in picture
        
        
        for x,y,w,h in carplate_rects:  ### gets four points of car number plate

        
                
            
            carplate_img = carplate_img_rgb[y+15:y+h-10 ,x+15:x+w-20] ### cutting car plate

            if len(carplate_img)> 0: ### Check car plate cutted correctly

                ### Some calculations, Magic))
                
                scale_per = 150
                w = int(carplate_img.shape[1] * scale_per / 100)
                h = int(carplate_img.shape[0] * scale_per / 100)
                dimension = (w, h)
                carplate_extract_img = cv2.resize(carplate_img, dimension, interpolation = cv2.INTER_AREA)           


                carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)

                carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray,3) # Character size 3

                ### Extracting text from car plate

                text = pytesseract.image_to_string(carplate_extract_img_gray_blur, config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                text = text[:-2]
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5)
                cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)

                ### IF user presses  "s" from keyboard. car number will be saved to scans folder

                if cv2.waitKey(1) & 0xFF ==ord('s'):  
                    
                    cv2.imwrite("scans/scan"+str(count)+".jpg",carplate_img)
                    
                    cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
                    cv2.putText(img,"Saved!",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)  ### informing user
                    
                    count += 1
                    
            
    cv2.imshow("Main",img)

    if cv2.waitKey(1) == 27:   ### Press "Esc" multiple times to kill program
        break


### Tugadi
    
cap.release()
cv2.destroyAllWindows()

