'''


In this script I made derive the friction velocity U* (Ustar) and plot it. I also normalize the fluctuating velocity components and plot them as well. 

Based on code written in ReynoldsDecomp0017 modified to read in data that has been extracted previously

Note: Look at field.py

'''
import matplotlib.pyplot as plt
import numpy as np
import sys
from tempfile import TemporaryFile

#LOAD DATA THAT HAS BEEN PREVIOUSLY EXTRACTED

#FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0005/'
FileDir = '/mnt/d/Documents/Brunel/Data/summary_rhouP_data/'

UData = np.load(FileDir+'UDataFile.npy')
PData = np.load(FileDir+'PDataFile.npy')
RhoData = np.load(FileDir+'RhoDataFile.npy')
X_Data = np.load(FileDir+'XCoordFile.npy',allow_pickle=True)

X=X_Data
rowsXCoord=np.zeros(3)
#Modify X-coordinates if there are negative coordinates

for index in range(3):
	rowsXCoord[index]=X[index].shape[0]
	if X[index][0]<0:
		CoordMax = int(rowsXCoord[index]-1)
		X[index][:]= (X[index][:]+X[index][CoordMax])*0.5
		
	
Xcoord = X[0]
Ycoord= X[1]
Zcoord = X[2]

#REYNOLDS DECOMPOSITION PORTION OF CODE

#User Controls

#0-206 for MD field
#30-55 for OF CFD at Re 2600
#30-50 for OF CFD at Re 400

StartTimeSample =0#start calculation from periods after turbulent field development
EndTimeSample=206

#Physical Constants
RhoVal=1.2 #default value for rho given in OF
Nu = 0.625 #kinematic viscosity, given by Ed

#Time Averaged Components
UBar = np.mean(UData[:,:,:,StartTimeSample:EndTimeSample,:], axis=3)
RhoBar = np.mean(RhoData[:,:,:,StartTimeSample:EndTimeSample,:], axis=3)
PBar = np.mean(PData[:,:,:,StartTimeSample:EndTimeSample,:], axis=3)

#Fluctuating Components
UPrime = UData.copy() #do a deep copy to get starting point 
RhoPrime = RhoData.copy() #do a deep copy to get starting point 
PPrime = PData.copy() #do a deep copy to get starting point 

# Subtract Ave from Instantaneous Data to get fluctating (Prime) values

NumTimeSamples = len(UData[1,1,1,:,1])
for TimeVal in range(NumTimeSamples):

    UPrime[:,:,:,TimeVal,:]= UPrime[:,:,:,TimeVal,:]-UBar
    PPrime[:,:,:,TimeVal,:]= PPrime[:,:,:,TimeVal,:]-PBar

# Calculate Shear Stress, and U star 

UBar1D = np.mean(UBar[:,:,:,0],axis=(2,0))

Tau1D=Nu*RhoVal*np.gradient(UBar1D,Ycoord,axis=0,edge_order=2)
Rho1D = np.mean(RhoData[:,:,:,StartTimeSample:EndTimeSample,:],axis=(3,2,0))


[rowsTau]=Tau1D.shape

HalfChannel=0.5*Ycoord[int(rowsXCoord[1]-1)]


Top_Tau0=Tau1D[rowsTau-1]
Bottom_Tau0=Tau1D[0]
Tau0_av=np.mean([Bottom_Tau0,Top_Tau0]) #compute average of top and bottom wall shear
Ustar = np.sqrt(Tau0_av/RhoVal)

Re=Ustar*HalfChannel/Nu

print("The Reynolds number based on Half Channel and Friction Velocity is:")
print(Re)

# Non-Dimensionalise fluctuting velocity
UPrime1D=np.mean(UPrime,axis=3)
uNorm=UPrime[:,:,:,StartTimeSample:EndTimeSample,:]/Ustar
u1Norm=UPrime[:,:,:,StartTimeSample:EndTimeSample,0]/Ustar
u2Norm=UPrime[:,:,:,StartTimeSample:EndTimeSample,1]/Ustar
u3Norm=UPrime[:,:,:,StartTimeSample:EndTimeSample,2]/Ustar



#Reynolds Stress Velocity Components-Instantaneous
u1u1=u1Norm**2
u1u2=u1Norm*u2Norm
u1u3=u1Norm*u3Norm
u2u1=u1u2
u2u2=u2Norm**2
u2u3=u2Norm*u3Norm
u3u1=u1u3
u3u2=u2u3
u3u3=u3Norm**2


#Reynolds Stress Velocity Components-Mean
u1u1Bar=np.mean(u1u1,axis=3)
u1u2Bar=np.mean(u1u2,axis=3)
u1u3Bar=np.mean(u1u3,axis=3)
u2u1Bar=u1u2Bar
u2u2Bar=np.mean(u2u2,axis=3)
u2u3Bar=np.mean(u2u3,axis=3)
u3u1Bar=u1u3Bar
u3u2Bar=u2u3
u3u3Bar=np.mean(u3u3,axis=3)



TKE_Norm = 0.5*(u1u1Bar+u2u2Bar+u3u3Bar)
u1prime_norm = np.sqrt(u1u1Bar)
u2prime_norm = np.sqrt(u2u2Bar)

u1prime_Norm1D=np.mean(u1prime_norm,axis=(2,0))
u2prime_Norm1D=np.mean(u2prime_norm,axis=(2,0))
TKE_Norm1D=np.mean(TKE_Norm,axis=(2,0))

#If anisotropic, uvbar will be negative value
#Turb_isotropy=np.mean(u1u2Bar,axis=(0,1,2))
#print(Turb_isotropy)

#Viscous Shear Stress
U_norm=UBar[:,:,:,0]/Ustar
Y_Norm = Ycoord/HalfChannel
X_Norm=X/HalfChannel

dUdy=np.gradient(U_norm,Y_Norm,axis=1,edge_order=2)
dUdy_1D=np.mean(dUdy,axis=(2,0))


#Andersson TKE Budget
P=-u1u2Bar*dUdy
P_1D = np.mean(P,axis=(2,0))

q2v=(u1u1+u2u2*u3u3)*u2Norm
q2v_bar=np.mean(q2v,axis=3)
T=-0.5*np.gradient(q2v_bar,Y_Norm,axis=1,edge_order=2)
T_1D = np.mean(T,axis=(2,0))

pNorm=PPrime/Tau0_av #divide pressure by shear stress to get non-dimensional value


pv=pNorm[:,:,:,StartTimeSample:EndTimeSample,0]*u2Norm
pvBar=np.mean(pv,axis=3)
PI=(-1)*np.gradient(pvBar,Y_Norm,axis=1,edge_order=2)
PI_1D = np.mean(PI,axis=(2,0))

dkdy= np.gradient(TKE_Norm,Y_Norm,axis=1,edge_order=2)
dk2dy2= np.gradient(dkdy,Y_Norm,axis=1,edge_order=2)
D=(1/Re)*dk2dy2
D_1D=np.mean(D,axis=(2,0))

duidxj=np.zeros(np.shape(u1u1Bar))

for i in range(2):
	for j in range(2):
		duidxj_1=np.gradient(uNorm[:,:,:,:,i],X_Norm[j],axis=j,edge_order=2)
		duidxj_2=duidxj_1**2
		duidxj=duidxj+np.mean(duidxj_2,axis=3)
eps=(1/Re)*duidxj
eps_1D=np.mean(eps,axis=(2,0))
		

#plot 1D Shear Stress (Fig 1 Andersson)
fig1, ax = plt.subplots()

ViscousShearStress = (1/Re)*dUdy_1D
TurbulentShearStress = (-1)*np.mean(u1u2Bar,axis=(2,0))
ax.plot(Y_Norm,ViscousShearStress,':',label='Viscous Shear Stress')
ax.plot(Y_Norm,TurbulentShearStress,'--',label='Turbulent Shear Stress')
ax.set_xlabel("Y-Coord")
ax.set_ylabel("Shear Stress")
ax.set_title(r'Shear Stress Distribution across Channel at $Re_\tau$ of '+str(int(Re)))

ax.legend()


#plot 1D TKE
fig2, ax = plt.subplots()

ax.plot(Y_Norm,TKE_Norm1D,label='k')
ax.plot(Y_Norm,u1prime_Norm1D,'--',label='u-fluctuating')
ax.plot(Y_Norm,u2prime_Norm1D,':',label='v-fluctuating')
ax.plot(Y_Norm,Tau1D,'-o',label='Shear Stress')
ax.set_xlabel("Y-Coord")
ax.set_ylabel("Norm. TKE")
ax.set_title(r'1D Channel TKE vs YCoord at $Re_\tau$ of '+str(int(Re)))

ax.legend()

#plot 1D TKE Budget
fig3, ax = plt.subplots(figsize=(14,2))# was 14,2
ax.plot(Y_Norm,P_1D,'--',label='P')
ax.plot(Y_Norm,T_1D,'--',label='T')
ax.plot(Y_Norm,PI_1D,'--',label='PI')
ax.plot(Y_Norm,D_1D,'--',label='D')
ax.plot(Y_Norm,(-1)*eps_1D,'--',label='-Eps')
#ax.set_ylim(top=3)
#ax.set_ylim(bottom=-2)

ax.set_xlabel("Y-Coord")
ax.set_ylabel("Energy")
ax.set_title(r'1D Channel TKE Budget vs YCoord at $Re_\tau$ of '+str(int(Re)))


ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

plt.show()





