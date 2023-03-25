'''


This code was based off of the old ReynoldsDecomp series of scripts I made. The purpose of this script is just to extract flowfield variables U,P,Rho,and X from MD or OpenFOAM simulations using pyDataView and same them into numpy arrays. This is done as I have found this process to take a long time, especially with fine mesh OpenFOAM cases/ 

'''
import matplotlib.pyplot as plt
import numpy as np
import sys
from tempfile import TemporaryFile

ppdir = '/mnt/d/Documents/Brunel/PythonCodes/pyDataView'
sys.path.append(ppdir)
import postproclib as ppl

normal =0
component=0
startrec=0
endrec=206 #time 500 to 55000 in CFD

#SET WORKING DIRECTORY TO EXTRACT DATA FROM
#fdir = '/mnt/d/Documents/CFD/CoutteFlowStudies/CouetteFlow0005/' #CFD files directory
fdir = '/mnt/d/Documents/Brunel/Data/summary_rhouP_data/'


PPObj = ppl.All_PostProc(fdir) #this class the allpostproc.py script which 


#GET DESIRED FIELD VARIABLES

#for CFD
#plotObj = PPObj.plotlist['U']
#plotObj2 = PPObj.plotlist['p']# Pressure Data (Note that this is kinemetic pessure)

#for MD
plotObj = PPObj.plotlist['u']# U data (all velocity components)
plotObj2 = PPObj.plotlist['P']# Pressure data
plotObj3 = PPObj.plotlist['rho']# Rho Data (use rho for now)



#SAVE FIELD VARIABLES INTO ARRAYS. 

RhoVal=1.2 #default value for rho given in OF

X = plotObj.grid[:] #Array of coordinate values X[X1,X2,X3] aka (X,Y,Z), 

#Use the Read method to get entire velocity feild
#CFD
#UData = plotObj.read(startrec=startrec,endrec=endrec) # UData[#X,#Y,#Z,#T,#Velocity components], note only velocity data given here, this is instantaneous velocity
#PData = RhoVal*plotObj2.read(startrec=startrec,endrec=endrec) # PData[#X,#Y,#Z,#T,Pressure values], note OF gives kinematic pressure so multiply by density
#RhoData = RhoVal*np.ones(PData[:,:,:,:,:].shape) # Density is constant but make an array of similar size to U and P Data

#MD
UData = plotObj.read(startrec=startrec,endrec=endrec) # UData[#X,#Y,#Z,#T,#Velocity components], note only velocity data given here, this is instantaneous velocity
PData =plotObj2.read(startrec=startrec,endrec=endrec) 
RhoData =plotObj3.read(startrec=startrec,endrec=endrec) 


#Save Data Arrays so we do not have to extract them again

np.save('UDataFile',UData)
np.save('PDataFile',PData)
np.save('RhoDataFile',RhoData)
np.save('XCoordFile',X)
