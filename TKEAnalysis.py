'''
Script to calculated TKE Budget for 1D Channel using formulation given in Anderrson paper. 
U values normalised by Ustar
Xcoord values normalised by HalfChannel
P vales normalised by Tau Wall

'''
import matplotlib.pyplot as plt
import numpy as np
import sys
from tempfile import TemporaryFile

#LOAD DATA THAT HAS BEEN PREVIOUSLY EXTRACTED
FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0007/'# Andersson
#FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0005/'# OF MD flow
#FileDir = '/mnt/d/Documents/Brunel/Data/summary_rhouP_data/'#MD Flow


#Upload Pre Calculated Data
UBarNorm = np.load(FileDir+'UBarNormFile.npy')
UPrimeNorm = np.load(FileDir+'UPrime_NormFile.npy')


PPrimeNorm = np.load(FileDir+'PPrimeNormFile.npy')

XNorm = np.load(FileDir+'XNormFile.npy', allow_pickle =True)
HalfChannel = np.load(FileDir+'HalfChannelFile.npy')
UStar = np.load(FileDir+'UstarFile.npy')
Re = np.load(FileDir+'ReynoldsNumberFile.npy')

print("Calculating....")

print("Reynolds number is:")
print(Re)

print("Ustar is:")
print(UStar)

print("HalfChannel is:")
print(HalfChannel)

#Break down coordinates

Xcoord =XNorm[0]
Ycoord =XNorm[1]
Zcoord =XNorm[2]

#Calculate 1D dU/Dy

UBarNorm1D = np.mean(UBarNorm, axis =(0,2))
dUdy=np.gradient(UBarNorm1D[:,0],Ycoord,axis=0,edge_order=2)

#Calculate u1u2, q, and k

u1 = UPrimeNorm[:,:,:,:,0]
u2 = UPrimeNorm[:,:,:,:,1]
u3 = UPrimeNorm[:,:,:,:,2]

u1u2 = u1*u2
q2= (u1**2 + u2**2 + u3**2)


k_1D= 0.5*np.mean(q2,axis = (3,0,2)) #mean TKE

#Calculate P, shear production
u1u2Bar = np.mean(u1u2, axis = (3,0,2))
P = -u1u2Bar*dUdy 

#Calculate T, turbulent diffusion associated with velocity
u2v_Bar = np.mean(u1**2*u2, axis = (3,0,2))
q2v_Bar = np.mean(q2*u2, axis = (3,0,2))
T = -0.5*np.gradient(q2v_Bar,Ycoord,axis=0,edge_order=2)

#Calculated pi, pressure fluctuations

pv_bar = np.mean(PPrimeNorm[:,:,:,:,0]*u2, axis = (3,0,2))
pi = np.gradient(pv_bar,Ycoord,axis=0,edge_order=2)

#Calculate D, Viscous Diffusion

dkdy = np.gradient(k_1D,Ycoord,axis=0,edge_order=2)
d2kdy2=np.gradient(dkdy,Ycoord,axis=0,edge_order=2)
D= -(1/Re)*d2kdy2

#Calculate eps, viscous energy dissipation

duidxj=np.zeros(np.shape(Ycoord))

for i in range(3):
	for j in range(3):
		duidxj_1=np.gradient(UPrimeNorm[:,:,:,:,i],XNorm[j],axis=j,edge_order=2)
		duidxj_2=duidxj_1**2
		duidxj=duidxj+np.mean(duidxj_2,axis=(3,0,2))
#duidxj_1=np.gradient(UPrimeNorm[:,:,:,:,0],XNorm[1],axis=1,edge_order=2)
#duidxj_2=duidxj_1**2
#duidxj=duidxj+np.mean(duidxj_2,axis=(3,0,2))


eps=(-1/Re)*duidxj


TurbVsDiss = P+eps



#plot 1D TKE Budget
fig1, ax = plt.subplots()# was 14,2
ax.plot(Ycoord,P,'-',label='P')
ax.plot(Ycoord,T,'--',label='T')
ax.plot(Ycoord,pi,'^-',label='PI')
ax.plot(Ycoord,D,'*-',label='D')
ax.plot(Ycoord,eps,'--',label='-Eps')


ax.set_xlabel("Y-Coord")
ax.set_ylabel("Energy")
ax.set_title(r'1D Channel TKE Budget vs YCoord at $Re_\tau$ of '+str(int(Re)))
ax.legend()
ax.set_ylim(bottom=-30)


#plot 1D Third order moments (Andersson fig 4)
fig2, ax = plt.subplots()


ax.plot(Ycoord,q2v_Bar,'-',label='q2v')
ax.plot(Ycoord,u2v_Bar,'--',label='u2v')
ax.set_xlabel("Y-Coord")
ax.set_ylabel("Third Order Moments")
ax.set_title(r'Third Order Moment Distribution across Channel at $Re_\tau$ of '+str(int(Re)))

ax.legend()



#plot 1D Turbulence generation vs dissipation
fig3, ax = plt.subplots()


ax.plot(Ycoord,TurbVsDiss,'-',label='Production+Dissipation')
ax.set_xlabel("Y-Coord")
ax.set_ylabel("Energy")
ax.set_title(r'Turbulence Production vs Dissipation Distribution across Channel at $Re_\tau$ of '+str(int(Re)))

ax.legend()

plt.show()




