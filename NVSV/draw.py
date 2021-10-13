import cv2
import numpy as np 
from matplotlib import pyplot as plt
import nn

drawing = False
last_x = None
last_y = None

def draw_line(event,x,y,flags,param):
    global last_x, last_y, drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        last_x,last_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing != None and drawing:
            cv2.line(img,(last_x,last_y),(x,y),color=(255),thickness=75)
            last_x,last_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(last_x,last_y),(x,y),color=(255),thickness=75)   

img = np.zeros((600,600,1), np.uint8)
cv2.namedWindow('draw')
cv2.setMouseCallback('draw',draw_line)


print('  _   _ _    _ __  __ ____  ______ _____    _____  ______ _______ ______ _____ _______ _____ ____  _   _ ')
print(' | \ | | |  | |  \/  |  _ \|  ____|  __ \  |  __ \|  ____|__   __|  ____/ ____|__   __|_   _/ __ \| \ | |')
print(' |  \| | |  | | \  / | |_) | |__  | |__) | | |  | | |__     | |  | |__ | |       | |    | || |  | |  \| |')
print(' | . ` | |  | | |\/| |  _ <|  __| |  _  /  | |  | |  __|    | |  |  __|| |       | |    | || |  | | . ` |')
print(' | |\  | |__| | |  | | |_) | |____| | \ \  | |__| | |____   | |  | |___| |____   | |   _| || |__| | |\  |')
print(' |_| \_|\____/|_|  |_|____/|______|_|  \_\ |_____/|______|  |_|  |______\_____|  |_|  |_____\____/|_| \_|')
print('---------------------------------------------------------------------------------------------------------')
print('[Enter] Predict Number | [C]lear | [Q]uit')
while(1):
    cv2.imshow('draw',img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    if key == ord('c'):
        img = np.zeros((600,600,1), np.uint8)
    if key == 13:
        image = cv2.resize(img,(28,28))
        prediction = nn.predict(image)
        max_index = np.argmax(prediction)
        print("It's a "+'\033[1m'+str(max_index)+'\033[0m')

cv2.destroyAllWindows()
    