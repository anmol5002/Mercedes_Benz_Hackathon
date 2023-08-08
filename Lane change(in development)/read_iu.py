import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas import DataFrame

import numpy

def readcsv(filename):
    data = pd.read_csv(filename) #Please add four spaces here before this line
    return(np.array(data)) #Please add four spaces here before this line


x = readcsv('q1.csv')
y = readcsv('s1.csv')
green=readcsv('s1_green.csv')
green_y=readcsv('q1_green.csv')
blue=readcsv('s1_blue.csv')
blue_y=readcsv('q1_blue.csv')
# plt.scatter(-y,x,color='r')
# plt.xlim(-30,30)
# plt.show()

# print(y)

i=[]
u=[]
i1=[]
i2=[]
u1=[]
u2=[]
# print(r)

for j in range(len(x)):
	i.append(x[j][0]-2.5)
	u.append(y[j][0])
	i1.append(green[j][0])
	i2.append(green_y[j][0]-2.5)
	u1.append(blue[j][0])
	u2.append(blue_y[j][0]-2.5)
# print(i)
# print(u)

m=[[1,0,0],[0,1,0],[0.02,0,1]]
m1=[[1,0,0],[0,1,0],[0.5,0,1]]
a=[]
b=[]
c=[]
a1=[]
b1=[]
c1=[]
a2=[]
b2=[]
c2=[]
for g in range(len(x)):
	a.append(m1[0][0]*i[g]+m1[0][1]*u[g]+m1[0][2])
	b.append(m1[1][0]*i[g]+m1[1][1]*u[g]+m1[1][2])
	c.append(m1[2][0]*i[g]+m1[2][1]*u[g]+m1[2][2])
	a1.append(m[0][0]*i1[g]+m[0][1]*i2[g]+m[0][2])
	b1.append(m[1][0]*i1[g]+m[1][1]*i2[g]+m[1][2])
	c1.append(m[2][0]*i1[g]+m[2][1]*i2[g]+m[2][2])
	a2.append(m[0][0]*u1[g]+m[0][1]*u2[g]+m[0][2])
	b2.append(m[1][0]*u1[g]+m[1][1]*u2[g]+m[1][2])
	c2.append(m[2][0]*u1[g]+m[2][1]*u2[g]+m[2][2])

# print(a)
# print(b)
# print(c)
i=[]
u=[]
i1=[]
i2=[]
u1=[]
u2=[]
for g in range(len(a)):
	i.append(["%.2f" % (a[g]/c[g])])
	u.append(["%.2f" % (b[g]/c[g])])
	i1.append(["%.2f" % (a1[g]/c1[g])])
	i2.append(["%.2f" % (b1[g]/c1[g])])
	u1.append(["%.2f" % (a2[g]/c2[g])])
	u2.append(["%.2f" % (b2[g]/c2[g])])




plt.scatter(i,u,color='r')
plt.scatter(i2,i1,color='g')
plt.scatter(u2,u1,color='b')
plt.xlim(-10,10)
plt.ylim(-10,75)
plt.show()

