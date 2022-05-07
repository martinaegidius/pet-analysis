import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class Overview():
    def __init__(self,doPlots=False):
        self.colorDict = {0:'firebrick',1:'lime',2:'b',3:'magenta',4:'c'}
        self.loadAllData()
        if doPlots:
            self.plotSomeData()
            # self.computeAreaThresholds()
            # self.testAndPlotOnLastThreeTotalArea()

    def loadAllData(self):
        path = 'Data/'
        patientDict = {}
        for i in range(1,11):
            patientDict[f'Patient {i}'] = pd.read_csv(path+f'patient{i}.csv',
                                                        header=None)  
            patientDict[f'Patient {i}'].iloc[0,:]=0

        self.patientDict = patientDict

    def computeAreaThresholds(self):
        healthy1 = self.patientDict['Patient 1'].copy()
        healthy2 = self.patientDict['Patient 2'].copy()
        healthy3 = self.patientDict['Patient 3'].copy()
        helth = [healthy1.iloc[:,2:],healthy2.iloc[:,2:],healthy3.iloc[:,2:]]
        sick1 = self.patientDict['Patient 4'].copy()
        sick2 = self.patientDict['Patient 5'].copy()
        sick3 = self.patientDict['Patient 6'].copy()
        sik = [sick1.iloc[:,2:],sick2.iloc[:,2:],sick3.iloc[:,2:]]

        for he in helth:
            for i, ye in enumerate(he):
                area = integrate.simpson(he.iloc[:,i])
                try:
                    areaList.append(area)
                except:
                    areaList = [area]
            try:    
                heList.append(np.sum(areaList))
                b = sorted(areaList)
                heTotList.append(b) #areaList.sort(reverse=True))
            except:
                heList = [np.sum(areaList)]
                b = sorted(areaList)
                heTotList =[b] # areaList.sort(reverse=True)]
            areaList=[]
        
        for si in sik:
            for i, sk in enumerate(si):
                area = integrate.simpson(si.iloc[:,i])
                try:
                    areaList.append(area)
                except:
                    areaList = [area]
            try:    
                siList.append(np.sum(areaList))
                b = sorted(areaList)
                siTotList.append(b) # areaList.sort(reverse=True))
            except:
                siList = [np.sum(areaList)]
                b = sorted(areaList)
                siTotList = [b] #areaList.sort(reverse=True)]
            areaList=[]

        self.areaThres = np.mean((np.mean(heList),np.mean(siList)))


    def testAndPlotOnLastThreeTotalArea(self):
        self.computeAreaThresholds()

        savepath = 'grafer/'
        # plt.style.use('seaborn')
        fig, axs = plt.subplots(1, 4)
        row = 0
        
        for i, key in enumerate(list(self.patientDict.keys())):
            if i <= 5:
                continue 
            col=i
            dat = self.patientDict[key]
            dat2=dat.copy()
            dat2=dat2.iloc[:,2:]
            for j, ye in enumerate(dat2):
                area = integrate.simpson(dat2.iloc[:,j])
                print('area er: ',area)
                try:
                    areaList.append(area)
                except:
                    areaList = [area]
            # areaList.sort()
            # patientRatio = areaList[-1]/areaList[0]

            patientThres = np.sum(areaList)
            print(key, patientThres,self.areaThres)
            areaList=[]
            # print(key,patientThres,self.areaThres)
            # print(key,patientRatio,self.ratioThres)
            # exit()
            
            # estimate = 'Healthy' if patientThres >= self.areaThres else 'Sick'
            estimate = 'Healthy' if patientThres >= self.areaThres else 'Sick'

            if i>2:
                row=0
                col-=6

            alp=1
            lwd=1.2
            # axs[col].plot(dat[0],dat[1],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[2],alpha=alp,linewidth=lwd,color=self.colorDict[0])
            axs[col].plot(dat[0],dat[3],alpha=alp,linewidth=lwd,color=self.colorDict[1])
            axs[col].plot(dat[0],dat[4],alpha=alp,linewidth=lwd,color=self.colorDict[2])
            axs[col].plot(dat[0],dat[5],alpha=alp,linewidth=lwd,color=self.colorDict[3])
            axs[col].plot(dat[0],dat[6],alpha=alp,linewidth=lwd,color=self.colorDict[4])
            axs[col].set_ylim(0,50) #,200)
            axs[col].set_title(f'{key}'+f' estimated as {estimate}',fontsize=12.5)
            axs[col].axhline(y=self.areaThres/1000,c='cyan',linestyle='-.',linewidth=1)
            axs[col].axhline(y=patientThres/1000,c='k',linestyle='-.',linewidth=1)

            colLabel = 'Activity [kBq/mL]' if col==0 else ''
            rowLabel = 'Time in minutes' #if row==1 else ''

            axs[col].set_xlabel(rowLabel,fontsize=14)
            axs[col].set_ylabel(colLabel,fontsize=14)

            fig.legend([#'Tracer activity\n in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4',
                    'Region 5','Threshold/1000','Mean for patient/1000'],fontsize=12, loc='right') #,'Threshold','Mean for patient'],
                    #fontsize=10)

        fig.set_size_inches((13,7),forward=False)
        fig.savefig(savepath+'AreathresholdVurdering.pdf',dpi=700,bbox_inches='tight')
        plt.show()

    def plotSomeData(self):
        savepath = 'grafer/'
        # plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 2)
        row = 0
        for i, key in enumerate(list(self.patientDict.keys())):
            col=i
            dat = self.patientDict[key]
            
            if i == 1:
                continue
            if i == 4:
                continue
            
            if i>1:
                
                col-=1
            if i>2:
                row=1
                col-=2
            if i > 4:
                col-=1
            
            lwd=1.2
            alp=1
            axs[row,col].plot(dat[0],dat[1],linewidth=lwd,alpha=alp)
            axs[row,col].plot(dat[0],dat[2],alpha=alp,linewidth=lwd,color=self.colorDict[0])
            axs[row,col].plot(dat[0],dat[3],alpha=alp,linewidth=lwd,color=self.colorDict[1])
            axs[row,col].plot(dat[0],dat[4],alpha=alp,linewidth=lwd,color=self.colorDict[2])
            axs[row,col].plot(dat[0],dat[5],alpha=alp,linewidth=lwd,color=self.colorDict[3])
            axs[row,col].plot(dat[0],dat[6],alpha=alp,linewidth=lwd,color=self.colorDict[4])
            axs[row,col].set_ylim(0,50)
            # axs[row,col].ticks_params(fontsize=14)
            axs[row,col].tick_params(axis = 'both', which = 'major', labelsize = 14)

            # axs[row,col].yticks(fontsize=14)

            axs[row,col].set_title(f'{key}',fontsize=14)
            if col==0:
                colLabel = 'Activity [kBq/mL]'
            else:
                colLabel=''
            if row==1:
                rowLabel = 'Time in minutes'
            else:
                rowLabel=''
            axs[row,col].set_xlabel(rowLabel,fontsize=20)
            axs[row,col].set_ylabel(colLabel,fontsize=20)

            if i >= 5:
                break
        fig.legend(['Tracer activity\n in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4','Region 5'],
                    fontsize=16,loc='upper right')
        fig.set_size_inches((13,7),forward=False)
        plt.savefig(savepath+'kunFireLabelledGrafer.pdf',dpi=700,bbox_inches='tight')
        plt.show()
            
    def nittyGrittyPlot(self):
        for key in list(self.patientDict.keys()):
            dat = self.patientDict[key]
            print(np.shape(dat[0]))
            print(np.shape(dat[1]))
            plt.plot(dat[0],dat[1])
            plt.plot(dat[0],dat[2],c='g')
            plt.plot(dat[0],dat[3],c='r')
            plt.plot(dat[0],dat[4],c='y')
            plt.plot(dat[0],dat[5],c='k')
            plt.legend(['Tracer activity in arterial blood','Region 1','Region 2','Region 3','Region 4','Region 5'])
            plt.title(f'{key}')
            plt.xlabel('Time in minutes')
            plt.ylabel('Activity')
            plt.show()


if __name__ == "__main__":
    Overview(doPlots=True)
