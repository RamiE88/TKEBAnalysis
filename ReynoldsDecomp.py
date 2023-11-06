#The purpose of this code is to compare instantaneous shear stress, mean stear stress, and the normalized values

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import copy
from tempfile import TemporaryFile

#LOAD DATA THAT HAS BEEN PREVIOUSLY EXTRACTED

FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0003/' #Andersson use averaging from 25 to 55
#FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0006/' #OF - Anderssen, more data
#FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0007/' #OF - Anderssen, lower courant and higher order interpolation
#FileDir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0005/' #OF - MD equivalent
#FileDir = '/mnt/d/Documents/Brunel/Data/summary_rhouP_data/' #0 to 206

UDataRaw = np.load(FileDir+'UDataFile.npy')
RhoDataRaw = np.load(FileDir+'RhoDataFile.npy')#for MD
PDataRaw = np.load(FileDir+'PDataFile.npy')#for MD
#RhoDataRaw = np.ones(UDataRaw.shape)# for Anderssen
X_Data = np.load(FileDir+'XCoordFile.npy',allow_pickle=True)
X=X_Data
rowsXCoord=np.zeros(3)
#Modify X-coordinates if there are negative coordinates

for index in range(3):
	rowsXCoord[index]=X[index].shape[0]
	if X[index][0]<0:
		CoordMax = int(rowsXCoord[index]-1)
		X[index][:]= (X[index][:]+X[index][CoordMax])
			
Xcoord = X[0]
Ycoord= X[1]
Zcoord = X[2]


#REYNOLDS DECOMPOSITION PORTION OF CODE

#User Controls

#0-206 for MD field
#30-55 for OF CFD at Re 2600
#30-50 for OF CFD at Re 400

StartTimeSample =30#start calculation from periods after turbulent field development
EndTimeSample=55
#Physical Constants
#Nu =.625 #MD case from ed .625
Nu = 0.107846154 #kinematic viscosity, Andersson case
#Nu=.701 #for OF Re=400 case

isMD =0

#MD data for fluid is at 2nd cell
if isMD:
	TopShift =2
	BotShift =1

else:

	TopShift =1
	BotShift =0

SaveVars=0
PlotData =1


print("Calculating...")


#Make Udata UBar and UPrime
UData=UDataRaw[:,:,:,StartTimeSample:EndTimeSample,:]

PData=PDataRaw[:,:,:,StartTimeSample:EndTimeSample,:]
#UData = UDataRaw
UData_copied = copy.deepcopy(UData)
RhoData=RhoDataRaw[:,:,:,StartTimeSample:EndTimeSample,:]
#RhoData=RhoDataRaw
RhoData_copied = copy.deepcopy(RhoData)


#Calculate UPrime and UBar

NumYVals=UData_copied.shape[1]
NumTimeSamples = UData_copied.shape[3]


#Make a 3D UBar and PBar
UBar = np.ones(UData_copied[:,:,:,0,:].shape) #Ignore time variable for shape
UProfile = np.mean(UData_copied,axis = (0,2,3))
UPrime = UData_copied

PBar = np.ones(PData[:,:,:,0,:].shape) #Ignore time variable for shape
PProfile = np.mean(PData,axis = (0,2,3))
PPrime = PData


for YVal in range(UData_copied.shape[1]):

	UBar[:,YVal,:,:]=UProfile[YVal]*UBar[:,YVal,:,:]
	PBar[:,YVal,:,:]=PProfile[YVal]*PBar[:,YVal,:,:]


# Calculate UPrime using 3D UBar
TimeArr = range(NumTimeSamples)

for TimeVal in TimeArr :
	UPrime[:,:,:,TimeVal,:]= np.subtract(UPrime[:,:,:,TimeVal,:],UBar)
	PPrime[:,:,:,TimeVal,:]= np.subtract(PPrime[:,:,:,TimeVal,:],PBar)


#Get Tau Wall, UStar, and Friction Reynolds Number

RhoBar1D=np.mean(RhoData_copied, axis =(0,2,3))


dUdy1D=np.gradient(UProfile[:,0],Ycoord,axis=0,edge_order=2)
mu_raw= Nu*RhoBar1D[:,0]

[rowsMu_raw] = mu_raw.shape

Top_mu_raw = mu_raw[rowsMu_raw-TopShift]# for MD dataset density is at second data point
Bottom_mu_raw = mu_raw[BotShift]

mu = np.mean([Top_mu_raw,Bottom_mu_raw])


Tau1D=mu*dUdy1D
[rowsTau]=Tau1D.shape

Top_Tau0=Tau1D[rowsTau-TopShift]
TopRho = RhoBar1D[rowsTau-TopShift]

Bottom_Tau0=Tau1D[BotShift]
BottomRho=RhoBar1D[BotShift]

Tau_Wall=np.mean([Bottom_Tau0,Top_Tau0]) #compute average of top and bottom wall shear
Rho_Wall=np.mean([BottomRho,TopRho])

Ustar = np.sqrt(Tau_Wall/Rho_Wall)
HalfChannel=0.5*Ycoord[int(rowsXCoord[1]-1)]
Re=Ustar*HalfChannel/Nu
Yplus = Ycoord*Ustar/Nu
XNorm = X/HalfChannel

PBarNorm = PBar/Tau_Wall
PPrimeNorm = PPrime/Tau_Wall

#Normalise Velocity and Ycoord

#Calc RMS velocity
Urms1D = np.sqrt(np.mean(UPrime[:,:,:,:,0]**2,axis =(0,2,3)))
Vrms1D = np.sqrt(np.mean(UPrime[:,:,:,:,1]**2,axis =(0,2,3)))
Wrms1D = np.sqrt(np.mean(UPrime[:,:,:,:,2]**2,axis =(0,2,3)))

TKE = 0.5*(UPrime[:,:,:,:,0]**2+UPrime[:,:,:,:,1]**2+UPrime[:,:,:,:,2]**2)
TKEDomain = np.sum(TKE,axis=(0,2,1))
UPrimeNorm = UPrime/Ustar
UBarNorm = UBar/Ustar
UBarNorm1D = UProfile/Ustar
YNorm = Ycoord/HalfChannel

#Find Normalised Fluctuating Velocity and uv

u1Norm = UPrimeNorm[:,:,:,:,0]
u2Norm = UPrimeNorm[:,:,:,:,1]
u3Norm = UPrimeNorm[:,:,:,:,2]

#Normalised Reynolds Stresses


u1u1Norm=u1Norm*u1Norm
u1u2Norm=u1Norm*u2Norm
u1u3Norm=u1Norm*u3Norm
u2u2Norm=u2Norm*u2Norm
u2u3Norm=u2Norm*u3Norm
u3u3Norm=u3Norm*u3Norm


#1D Normalised Reynolds Stresses

u1u1Norm1DBar = np.mean(u1u1Norm,axis=(3,0,2))
u1u2Norm1DBar = np.mean(u1u2Norm,axis=(3,0,2))
u1u3Norm1DBar = np.mean(u1u3Norm,axis=(3,0,2))
u2u2Norm1DBar = np.mean(u2u2Norm,axis=(3,0,2))
u2u3Norm1DBar = np.mean(u2u3Norm,axis=(3,0,2))
u3u3Norm1DBar = np.mean(u3u3Norm,axis=(3,0,2))


TurbulentShear1D = (-1)*u1u2Norm1DBar

#Find Viscous Shear
dUBardyNorm1D = np.gradient(UBarNorm1D[:,0],YNorm,axis=0)
ViscousShear1D = (1/Re)*dUBardyNorm1D


CombinedShear1D = ViscousShear1D+TurbulentShear1D

print("Calculation over.")

if PlotData:


	#plot 1D Shear Stress (Fig 1 Andersson)
	fig1, ax = plt.subplots()
	ax.plot(YNorm,ViscousShear1D,':',label='Viscous Shear Stress')
	ax.plot(YNorm,TurbulentShear1D,'--',label='Turbulent Shear Stress')
	ax.plot(YNorm,CombinedShear1D,'+',label='Combined Shear Stress')
	ax.set_xlabel("Y-Coord")
	ax.set_ylabel("Shear Stress")
	ax.set_title(r'Shear Stress Distribution across Channel at $Re_\tau$ of '+str(int(Re)))
	#ax.set_xlim(0,0.5)
	ax.legend()



	fig2, ax = plt.subplots()
	ax.plot(TimeArr,TKEDomain,'--',label='Viscous Shear Stress')
	ax.set_xlabel("Time Step")
	ax.set_ylabel("Domain TKE")
	ax.set_title(r'Average TKE in Domain at $Re_\tau$ of '+str(int(Re)))





	#plot 1D Shear Stress (Fig 1 Andersson) - Yplus
	fig3, ax = plt.subplots()
	ax.plot(Yplus,ViscousShear1D,'k-o',label='Viscous Shear Stress')
	ax.plot(Yplus,TurbulentShear1D,'-o',label='Turbulent Shear Stress')
	ax.plot(Yplus,CombinedShear1D,'+',label='Combined Shear Stress')
	ax.set_xlabel("Y-Plus")
	ax.set_ylabel("Shear Stress")
	ax.set_title(r'Shear Stress Distribution across Channel at $Re_\tau$ of '+str(int(Re)))
	ax.set_xlim(0,30)

	ax.legend()


	#Ed's RMS plot
	fig4, ax = plt.subplots()
	ax.plot(YNorm,Urms1D,'k-',label='Urms')
	ax.plot(YNorm,Vrms1D,'k--',label='Vrms')
	ax.plot(YNorm,Wrms1D,'k.',label='Wrms')
	ax.set_xlabel("y/H")
	ax.set_ylabel("Urms Vrms Wrms")
	ax.set_xlim(0,1)
	ax.legend()

	#Reynolds Stress plot
	fig5, ax = plt.subplots()
	ax.plot(YNorm,u1u1Norm1DBar,label='U1U1')
	ax.plot(YNorm,-u1u2Norm1DBar,label='U1U2')
	ax.plot(YNorm,u1u3Norm1DBar,label='U1U3')
	ax.plot(YNorm,u2u2Norm1DBar,label='U2U2')
	ax.plot(YNorm,u2u3Norm1DBar,label='U2U3')
	ax.plot(YNorm,u3u3Norm1DBar,label='U3U3')
	ax.set_title(r'Reynolds Stress Components across Channel at $Re_\tau$ of '+str(int(Re)))
	ax.set_xlabel("y/H")
	ax.set_ylabel("Stress")
	ax.set_xlim(0,1)
	ax.legend()





	plt.show()


if SaveVars:

	#Save all data that can be used for TKE analysis

	np.save(os.path.join(FileDir,'UPrimeFile'),UPrime)
	np.save(os.path.join(FileDir,'UPrime_NormFile'),UPrimeNorm)
	
	np.save(os.path.join(FileDir,'UBarFile'),UBar)
	np.save(os.path.join(FileDir,'UBarNormFile'),UBarNorm)
	
	np.save(os.path.join(FileDir,'UstarFile'),Ustar)
        
	np.save(os.path.join(FileDir,'XNormFile'),XNorm)
	np.save(os.path.join(FileDir,'HalfChannelFile'),HalfChannel)

	np.save(os.path.join(FileDir,'PBarNormFile'),PBarNorm)
	np.save(os.path.join(FileDir,'PPrimeNormFile'),PPrimeNorm)
	np.save(os.path.join(FileDir,'TauWallFile'),Tau_Wall)
	np.save(os.path.join(FileDir,'ReynoldsNumberFile'),Re)


	print("Data Saved.")
	




