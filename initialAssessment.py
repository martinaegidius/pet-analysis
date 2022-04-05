#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:31:58 2022

@author: max
"""

import pandas as pd 
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

path = "Data/"
fileList = sorted(glob.glob(os.path.join(path,"*.csv")))
hold = fileList.pop(1) #maintain order
fileList.append(hold) 


#patientDict = {}
data = np.empty((len(fileList),158,7))

for i, filename in enumerate(fileList):
    print(filename)
    df = pd.read_csv(str(filename),header=None)
    data[i] = np.array(df)

'''for i in range(len(data)):  #all plots
   
    plt.title(fileList[i])
    plt.legend(["Arterial blood","Region 1","Region 2", "Region 3", "Region 4", "Region 5"])
    plt.xlabel("Minutes after injection")
    plt.ylabel("Activity [kBq/ml]")
    plt.show()
   ''' 
    
#healthy vs sick 
fig, axs = plt.subplots(2,3)
for i in range(6):
    if(i<3):
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,1].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,2].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,3].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,4].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,5].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,6].squeeze())
        axs[0,i].set_title("Patient "+str(i+1))
        axs[0,i].set_ylim([0,50])
    else: 
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,1].squeeze())
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,2].squeeze())
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,3].squeeze())
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,4].squeeze())
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,5].squeeze())
        axs[1,i-3].plot(data[i,:,0].squeeze(),data[i,:,6].squeeze())
        axs[1,i-3].set_title("Patient " + str(i+1))
        axs[1,i-3].set_ylim([0,50])
    fig.legend(["Arterial blood","Region 1","Region 2", "Region 3", "Region 4", "Region 5"])
    

#pooling healthy and sick to find mean curves 
sick = np.array([data[3],data[4],data[5]])
healthy = np.array([data[0],data[1],data[2]])
sickMeans = sick.mean(0)
healthyMeans = healthy.mean(0)
fig, axs = plt.subplots(1,2,sharey=True)
for i in range(2):
    if(i<1):
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,1].squeeze())
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,2].squeeze())
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,3].squeeze())
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,4].squeeze())
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,5].squeeze())
        axs[i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,6].squeeze())
        axs[i].set_title("Mean healthy")
    else: 
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,1].squeeze())
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,2].squeeze())
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,3].squeeze())
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,4].squeeze())
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,5].squeeze())
        axs[i].plot(sickMeans[:,0].squeeze(),sickMeans[:,6].squeeze())
        axs[i].set_title("Sick means")
    
    fig.legend(["Arterial blood","Region 1","Region 2", "Region 3", "Region 4", "Region 5"])
    
#healthy vs sick with mean curves
fig, axs = plt.subplots(2,4)
for i in range(8):
    if(i<3):
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,1].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,2].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,3].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,4].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,5].squeeze())
        axs[0,i].plot(data[i,:,0].squeeze(),data[i,:,6].squeeze())
        axs[0,i].set_title("Patient "+str(i+1))
        axs[0,i].set_ylim([0,50])
    elif(i==3):
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,1].squeeze())
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,2].squeeze())
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,3].squeeze())
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,4].squeeze())
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,5].squeeze())
        axs[0,i].plot(healthyMeans[:,0].squeeze(),healthyMeans[:,6].squeeze())
        axs[0,i].set_title("Mean healthy")
        axs[0,i].set_ylim([0,50])
    elif(i<7): 
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,1].squeeze())
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,2].squeeze())
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,3].squeeze())
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,4].squeeze())
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,5].squeeze())
        axs[1,i-4].plot(data[i,:,0].squeeze(),data[i,:,6].squeeze())
        axs[1,i-4].set_title("Patient " + str(i+1))
        axs[1,i-4].set_ylim([0,50])
    else: 
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,1].squeeze())
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,2].squeeze())
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,3].squeeze())
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,4].squeeze())
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,5].squeeze())
        axs[1,i-4].plot(sickMeans[:,0].squeeze(),sickMeans[:,6].squeeze())
        axs[1,i-4].set_title("Sick means")
        axs[1,i-4].set_ylim([0,50])
    fig.legend(["Arterial blood","Region 1","Region 2", "Region 3", "Region 4", "Region 5"])




