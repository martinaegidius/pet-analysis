import warnings
warnings.filterwarnings("ignore")

from overview import Overview

from scipy import integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Integration():
    def __init__(self):
        OVInstance = Overview()
        self.patientDict = OVInstance.patientDict
        # self.leastSquare()
        self.makeKforAll()
        self.plotKVals()
        # self.makeMeanofLeastSquareAndDefineK()
        # self.doSimpsons()
        # print(self.patientDict)

    def makeK(self,p):
        p1=p[0]
        p2=p[1]
        p3=p[2]
        p4=p[3]
        k1 = p1
        k2 = (p1*p3 - p2)/p1
        k3 = -(p1**2*p4 - p1*p2*p3 + p2**2)/((p1*p3 - p2)*p1)
        k4 = p4*p1/(p1*p3 - p2)
        return k1,k2,k3,k4


    def makeKforAll(self):
        newPatientDict={}
        for patient in self.patientDict.keys():
            for i in range(1,6):
                pDataFrame = pd.Series(self.leastSquare(region=i,patient=patient))
                pDataFrame = pDataFrame.T
                k1,k2,k3,k4 = self.makeK(pDataFrame)
                try:
                    newPatientDict[patient].append([k1,k2,k3,k4])
                except:
                    newPatientDict[patient] = [[k1,k2,k3,k4]]


        # print(newPatientDict['Patient 1'])
        # # print(newPatientDict)
        # for patient in newPatientDict.keys():
        #     print(patient)
        #     print(newPatientDict[patient])
        self.kDict = newPatientDict
   
    def plotKVals(self):
        # # dat = self.kDict['Patient 1']
        # print(dat[0])
        # print(dat[1])
        colorDict = {0:'r',1:'g',2:'b',3:'y',4:'c'}
        markerDict = {0:'^' ,1:'*' ,2:'o', 3:'s'}
        # plt.style.use('classic')
        fig, ax = plt.subplots()
        for h, patient in enumerate(self.kDict.keys()):
            for i,region in enumerate(self.kDict[patient]):
                for j, k in enumerate(region):
                    if j==0:
                        ax.axvline(x=h+(j/5.0)-0.1)
                    if (j==3):
                        print('hello')
                        if ('10' in patient):
                            print('yeehaw')
                            ax.axvline(x=h+(j/5.0)+0.1)
                    ax.scatter(h+(j/5.0),k,color=colorDict[i],marker=markerDict[j],alpha=0.5)

        ax.set_xticks([x+0.5 for x in list(range(len(self.kDict.keys())))])
        ax.set_xticklabels(list(self.kDict.keys()),rotation=30)

        # plt.grid('y') #,color = 'green', linestyle = '--', linewidth = 0.3)
        plt.show()  

   
    def makeMeanofLeastSquareAndDefineK(self):

        
        for i in range(1,6):
            try:
                pDataFrame = pd.concat((pDataFrame,pd.Series(self.leastSquare(region=i))),axis=1,ignore_index=True)
            except:
                pDataFrame = pd.Series(self.leastSquare(region=i))

        pDataFrame = pDataFrame.T
        print(pDataFrame)
        p = pDataFrame.mean()
        p1=p[0]
        p2=p[1]
        p3=p[2]
        p4=p[3]
        k1 = p1
        k2 = (p1*p3 - p2)/p1
        # k3 = -(p1**2*p4 - p1*p2*p3 + p2**2)/((p1*p3 - p2)*p1)
        k3 = -(pow(p1,2)*p4 - p1*p2*p3 + pow(p2,2))/((p1*p3 - p2)*p1)
        k4 = p4*p1/(p1*p3 - p2)

        print(k1,k2,k3,k4)


    def leastSquare(self,patient=False,region=1):
        if not patient:
            patient = 'Patient 1'

        A = self.doSimpsons(patient)
        b = self.patientDict[patient][region+1]
        b.drop(b.index[-1],inplace=True)
        k = np.linalg.lstsq(A, b)
        return k[0]

    def doSimpsons(self,patient=False,region=1):
        if not patient:
            patient = 'Patient 1'

        dat = self.patientDict[patient]
        col1=[]
        col2=[]
        col3=[]
        col4=[]

        for i in range(len(dat[0])):
            if i > 0:
                col1.append(integrate.simpson(dat[1][:i], x=dat[0][:i],even='last'))
                col2.append(integrate.simpson(col1[:i],x=dat[0][:i],even='last'))
                col3.append(-integrate.simpson(dat[region+1][:i], x=dat[0][:i],even='last'))
                col4.append(integrate.simpson(col3[:i],x=dat[0][:i],even='last'))

        totFrame = pd.concat((pd.Series(col1),pd.Series(col2),pd.Series(col3),pd.Series(col4)),ignore_index=True,axis=1)

        return totFrame



if __name__=='__main__':
    Integration()