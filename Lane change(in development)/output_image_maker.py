from PIL import Image
import numpy
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas import DataFrame

import numpy

def readcsv(filename):
    data = pd.read_csv(filename) #Please add four spaces here before this line
    return(np.array(data))
x = readcsv('prob.csv')
print(x)
iii=[]
for j in range(len(x)):
	iii.append(x[j][0])

print(iii)
img = Image.open('Figure_1.png') # image extension *.png,*.jpg
new_width  = 1600
new_height = 900
img = img.resize((new_width, new_height), Image.ANTIALIAS)
 # format may what u want ,*.png,*jpg,*.gif
img.save('Figure_1.png')
 

# Opens a image in RGB mode 
im = Image.open("Figure_1.png") 
  
# Setting the points for cropped image 
left = 190
top = 102
right = 1440
bottom = 750
  
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
im1 = im1.resize((new_width, new_height), Image.ANTIALIAS)
im1.save("b.png")
# Shows the image in image viewer 
# im1.show() 
img=cv2.imread('b.png')
print(img[0][0][0])
plt.imshow(img)
plt.show()

img1 = cv2.imread('b.png')
img2 = cv2.imread('n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800381862460.jpg')

dst = cv2.addWeighted(img1,0.2,img2,0.8,0)

plt.imshow(dst[:,:,::-1])
plt.show()
im_pil = Image.fromarray(dst[:,:,::-1])


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

blank_image = im_pil
img_draw = ImageDraw.Draw(blank_image)
font = ImageFont.truetype('arial',size=40)
img_draw.rectangle((70, 50, 450, 200), outline='red', fill='white')
img_draw.text((70, 60), str("%.2f" %(iii[1]/(iii[0]+iii[1]+iii[2])))+'  Danger path', fill='blue',font=font)
img_draw.text((70, 100), str("%.2f" %(iii[0]/(iii[0]+iii[1]+iii[2])))+'  Collision path', fill='red',font=font)
img_draw.text((70, 140), str("%.2f" %(iii[2]/(iii[0]+iii[1]+iii[2])))+'  Safe path', fill='green',font=font)
blank_image.save('drawn_image.jpg')