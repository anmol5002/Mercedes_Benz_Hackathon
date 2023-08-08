import matplotlib.pyplot as plt
import numpy as np
import math
n=np.arange(0,4,0.1)
m=np.arange(0,4,0.1)

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

x=[]
y=[]
z=[]
mlat=[]
mlat_ulta=np.arange(5.6,6.52,0.1)
t_ulta=np.arange(2.5,7.01,0.5)
# mlat_ulta=np.arange(5.6,6.520,0.1)
# t_ulta=np.arange(2.5,3.501,0.5)
x1=[]
y1=[]
x2=[]
y2=[]
for i in range(len(n)):
	for j in range(len(m)):
		if n[i]*n[i]-4*m[j]<0:
			if 5.5<=5.5+3/(math.exp(3.14*n[i]/math.sqrt(4*m[j]-n[i]*n[i])))<6.5:
				x.append(n[i])
				y.append(m[j])
				z.append(2*3.14/math.sqrt(4*m[j]-n[i]*n[i]))
				mlat.append(5.5+3/(math.exp(3.14*n[i]/math.sqrt(4*m[j]-n[i]*n[i]))))

for i in range(len(mlat_ulta)):
	for j in range(len(t_ulta)):
		if 2*mlat_ulta[i]+t_ulta[j]>14.2:
			x1.append(mlat_ulta[i])
			y1.append(t_ulta[j])

for i in range(len(x1)):
		x21=(2*math.log(3/(x1[i]-5.45)))/y1[i]
		y21=((x21*x21)+((2*3.14)/y1[i]*(2*3.14)/y1[i]))/4
		x2.append(x21)
		y2.append(y21)

# new model


print(x2)
# print(y2)
plt.scatter(y2,x2)

# plt.xlim(2,7)
# plt.ylim(5.5,6.5)
# print(gap)
plt.show()
q1=[[2.5 for _ in range(1)] for _ in range(len(y2))] 
s1=[[-3.33 for _ in range(1)] for _ in range(len(y2))] 
q2=[[2.50 for _ in range(1)] for _ in range(len(y2))]
s2=[[50 for _ in range(1)] for _ in range(len(y2))] 
dq1=[[0 for _ in range(1)] for _ in range(len(y2))] 
ds1=[[16.66 for _ in range(1)] for _ in range(len(y2))]
dq2=[[0 for _ in range(1)] for _ in range(len(y2))]
ds2=[[0 for _ in range(1)] for _ in range(len(y2))] 
gap=[[math.sqrt((q1[0][0]-q2[0][0])*(q1[0][0]-q2[0][0])+(s1[0][0]-s2[0][0])*(s1[0][0]-s2[0][0])) for _ in range(1)] for _ in range(len(y2))] 
ddq2=[[0 for _ in range(1)] for _ in range(len(y2))] 
dds2=[[0 for _ in range(1)] for _ in range(len(y2))] 
t=[[0 for _ in range(1)] for _ in range(len(y2))] 
min_gap=[]
for iu_x in range(len(y2)):
	for iu_y in range(1):
		dt=0.2
		for i in range(25):
			q1[iu_x].append(q1[iu_x][i]+dq1[iu_x][i]*dt)
			s1[iu_x].append(s1[iu_x][i]+ds1[iu_x][i]*dt)
			q2[iu_x].append(q2[iu_x][i]+dq2[iu_x][i]*dt)
			s2[iu_x].append(s2[iu_x][i]+ds2[iu_x][i]*dt)
			dq1[iu_x].append(dq1[iu_x][i]+(y2[iu_x]*(5.5-q1[iu_x][i]))-(x2[iu_x]*dt*dq1[iu_x][i]))
			ds1[iu_x].append(ds1[iu_x][i])
			dq2[iu_x].append(dq2[iu_x][i]+ddq2[iu_x][i]*dt)
			ds2[iu_x].append(ds2[iu_x][i]+dds2[iu_x][i]*dt)
			dds2[iu_x].append(0)
			ddq2[iu_x].append(0)
			t[iu_x].append(i*dt)
	
# plt.scatter(t[0],q1[0],c='b')
for i in range(len(y2)):
	for j in range(1,24):
		gap[i].append(calculateDistance(s1[i][j], q1[i][j], s2[i][j], q2[i][j]))
	min_gap.append(min(gap[i]))
# print(q1[0])
# print(q2[0])
redder_x=[]
redder_y=[]
green_x=[]
green_y=[]
bluer_x=[]
bluer_y=[]
red_color=0
blue_color=0
green_color=0
iu=0
ii=0
l=0
m=0
v=0

for iu_x in range(len(y2)):
	if min_gap[iu_x]<=2 and 0<y2[iu_x]<=0.7 and 0<=x2[iu_x]<1:
		redder_x.append(y2[iu_x])
		redder_y.append(x2[iu_x])
		l=l+1
		if iu==0:
			red_color=iu_x
			iu=iu+1
	elif 2<min_gap[iu_x]<=2.5 and 0<y2[iu_x]<=0.7 and 0<=x2[iu_x]<1:
		bluer_x.append(y2[iu_x])
		bluer_y.append(x2[iu_x])
		m=m+1
		if ii==0:
			blue_color=iu_x
			ii=ii+1
	elif min_gap[iu_x]>2.5 and 0<y2[iu_x]<=0.7 and 0<=x2[iu_x]<1:
		green_x.append(y2[iu_x])
		green_y.append(x2[iu_x])
		green_color=iu_x
		v=v+1
plt.scatter(s1[red_color], q1[red_color],color='red')
plt.scatter(s1[green_color], q1[green_color],color='g')
plt.scatter(s1[blue_color], q1[blue_color],color='b')
# plt.scatter(s1[64], q1[64],color='cyan')

# for i in range(len(redder_x)):
# 	red_x[i].append(s1[red_color[i]])
# 	red_y[i].append(q1[red_color[i]])
# 	plt.scatter(red_x[i],red_y[i])

a = np.asarray(s1[red_color])
np.savetxt("s1.csv", a)
a = np.asarray(q1[red_color])
np.savetxt("q1.csv", a)
a = np.asarray(s1[green_color])
np.savetxt("s1_green.csv", a)
a = np.asarray(q1[green_color])
np.savetxt("q1_green.csv", a)
a = np.asarray(s1[blue_color])
np.savetxt("s1_blue.csv", a)
a = np.asarray(q1[blue_color])
np.savetxt("q1_blue.csv", a)
a=[0,l,m,v]
np.savetxt('prob.csv',a)
# print(q1[15]) 

# print(min_gap)
print(gap[1])

# print()
# print(gap)
plt.show()

plt.scatter(redder_x, redder_y,color='red')
plt.scatter(green_x, green_y,color='g')
plt.scatter(bluer_x, bluer_y,color='b')
# print(x1)
# print(y1)

# plt.scatter(t[1][1],q1[1][1],c='r')
plt.xlim(0,2.5)
plt.ylim(0.2,2)
print(y2[1])
print(x2[1])
print(y2[2])
print(x2[2])
print(y2[64])
print(x2[64])
# print(gap)
# print(q1[15]) 
plt.show()