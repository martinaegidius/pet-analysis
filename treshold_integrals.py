import scipy.integrate as spi
import pandas as pd 
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
plt.style.use("seaborn")

#load in data
path = "Data/"
fileList = sorted(glob.glob(os.path.join(path,"*.csv")))
hold = fileList.pop(1) #maintain order
fileList.append(hold) 
data = np.empty((len(fileList),158,7))
for i, filename in enumerate(fileList):
    df = pd.read_csv(str(filename),header=None)
    data[i] = np.array(df)
data[:,0,:] = 0 #one solution to zero'th row corrupted in few measurements and adjusted to be = 0.


def regionWiseIntegral(region,data):
    areas = []
    r = region + 1
    print("integrating region {}".format(r))
    for i in range(data.shape[0]):
        areas.append(spi.simpson(data[i,:,r],x=data[i,:,0],even='avg'))
        print("appended patient {} with area {}".format(i+1,areas[i]))
                
    return areas

#Treshold-estimates
def getTreshold(values,num_health,num_sick): #expects data-structure Mpatients X Nregions with first num_health healthy individuals, and following num_sick sick individuals
    healthyMeans = np.mean(values[0:num_health,:],axis=0)
    sickMeans = np.mean(values[num_health:num_health+num_sick,:],axis=0)
    return (healthyMeans+sickMeans)/2,healthyMeans,sickMeans



#get all integrals 
areas = np.empty((data.shape[0],data.shape[2]-2))
for r in range(data.shape[2]-2):
    print(r)
    areas[:,r] = regionWiseIntegral(r,data)

#get treshold-classifier
theta,healthyMeans,sickMeans = getTreshold(areas,3,3)

def classifierPlot(data,healthyMeans,sickMeans):
    labels = ['R1','R2','R3','R4','R5']
    x = np.arange(1,len(labels)+1)
    width = 0.4
    fig = plt.figure()
    for i in range(data.shape[2]-2):
        if(i==0):
            plt.bar(x[i]-width/2,healthyMeans[i],width,label='Healthy',color='darkcyan')
            plt.bar(x[i]+width/2,sickMeans[i],width,label='Sick',color='coral')
            plt.hlines(y=theta[i],xmin=x[i]-width,xmax=x[i]+width,color='fuchsia',label='Treshold')
        else:
            plt.bar(x[i]-width/2,healthyMeans[i],width,color='darkcyan')
            plt.bar(x[i]+width/2,sickMeans[i],width,color='coral')
            plt.hlines(y=theta[i],xmin=x[i]-width,xmax=x[i]+width,color='fuchsia')
    plt.figlegend(ncol=3,loc='upper center',bbox_to_anchor=(0.5,0.95),fontsize=12)
    plt.xticks([1,2,3,4,5],labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Mean accumulated tracer [1/mL]',fontsize = 14)
    fig.set_size_inches((10,7),forward=False)
    plt.savefig('skrt3.pdf',dpi=700,bbox_inches='tight')
    plt.show()
    

classifierPlot(data,healthyMeans,sickMeans)

def classifyPlot(data,theta,num_health,num_sick):
    labels = ['Pt.7','Pt.8','Pt.9','Pt. 10']
    x = np.arange(1,len(labels)+1)
    width = 0.5
    fig = plt.figure()
    mean = theta.mean()
    patient_means = areas.mean(axis=1)
    for i in range(4):
        if(i==0):
            plt.bar(x[i],patient_means[i+6],width,color='coral',label='Categorized sick')
            plt.hlines(y=mean,xmin=0.7,xmax=4.3,color='fuchsia',label='Threshold',linewidth=2)
        elif(i==3):
            plt.bar(x[i],patient_means[i+6],width,color='darkcyan',label="Categorized healthy")
        else:
            plt.bar(x[i],patient_means[i+6],width,color='coral')
            #plt.hlines(y=mean,xmin=x[i]-width,xmax=x[i]+width,color='magenta',linestyles='dotted')
    plt.xticks([1,2,3,4],labels, fontsize=18,weight='bold')
    plt.yticks(fontsize=20,weight='bold')
    plt.ylabel(r'Mean accumulated tracer [1/mL]',fontsize=20,weight='bold') #($\frac{kBq}{mL\cdot s}$)'
    plt.figlegend(ncol=3,loc='upper center',bbox_to_anchor=(0.5,1),fontsize=17)
    fig.set_size_inches((10,7),forward=False)
    plt.savefig('skrt1.pdf',dpi=700,bbox_inches='tight')
    plt.show()
        
classifyPlot(areas,theta,3,3)

def classifyPatient(areas,theta):
    if(areas.mean()<=theta.mean()):
        return 1
    else:
        return 0
    
label = []
for i in range(areas.shape[0]):
    label.append(classifyPatient(areas[i,:],theta))
    if(label[i]==0):
        print("Patient {} is healthy".format(i+1))
    elif(label[i]==1):
        print("Patient {} is sick".format(i+1))
        
def classifyAllPlot(data,theta):
    labels = ['Pt.1','Pt.2','Pt.3','Pt.4','Pt.5','Pt.6']
    x = np.arange(1,len(labels)+1)
    width = 0.5
    fig = plt.figure()
    mean = theta.mean()
    patient_means = areas.mean(axis=1)
    for i in range(6):
        if(i==0):
            plt.bar(x[i],patient_means[i],width,color='darkcyan',label='Labelled healthy')
            plt.hlines(y=mean,xmin=0.7,xmax=6.3,color='fuchsia',label='Threshold',linewidth=2)
        elif(i<3):
            plt.bar(x[i],patient_means[i],width,color='darkcyan')
        elif(i==3):
            plt.bar(x[i],patient_means[i],width,color='coral',label='Labelled sick')
        else:
            plt.bar(x[i],patient_means[i],width,color='coral')
           
            #plt.hlines(y=mean,xmin=x[i]-width,xmax=x[i]+width,color='magenta',linestyles='dotted')
    plt.xticks([1,2,3,4,5,6],labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Mean accumulated tracer [1/mL]',fontsize=14)
    plt.figlegend(ncol=3,loc='upper center',bbox_to_anchor=(0.55,0.97),fontsize=12)
    fig.set_size_inches((10,7),forward=False)
    plt.savefig('skrt2.pdf',dpi=700,bbox_inches='tight')
    plt.show()
    plt.show()
        

    
    
classifyAllPlot(areas,theta)
    
    
        
