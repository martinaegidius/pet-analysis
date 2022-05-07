import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy 



class Overview():
    def __init__(self,doPlots=False):
        self.loadAllData()
        if doPlots:
            # self.plotSomeDataWithMeans()
            # self.testAndPlotOnLastThree()
            self.plotStackedPlot()

    def loadAllData(self):
        path = 'Data/'
        patientDict = {}
        for i in range(1,11):
            patientDict[f'Patient {i}'] = pd.read_csv(path+f'patient{i}.csv',
                                                        header=None)        
        self.patientDict = patientDict

    def plotStackedPlot(self):
        savepath = 'grafer/'
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 3)
        row = 0


        for i, key in enumerate(list(self.patientDict.keys())):
            
            col=i
            dat = self.patientDict[key]
            
            if i>2:
                row=1
                col-=3

            axs[row,col].stackplot(dat[0],dat[1],dat[2],dat[3],dat[4],dat[5],dat[6],
                labels=['Region 1','Region 2','Region 3','Region 4','Region 5','Region 6'])
            axs[row,col].set_title(f'{key}',fontsize=12)
            axs[row,col].set_ylim(0,250) #,200)
            colLabel = 'Activity' if col==0 else ''
            rowLabel = 'Time in minutes' if row==1 else ''
            axs[row,col].set_xlabel(rowLabel,fontsize=12)
            axs[row,col].set_ylabel(colLabel,fontsize=12)
            if i >= 5:
                break

        fig.legend(['Tracer activity\n in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4',
                    'Region 5'],
                    fontsize=12)
        plt.savefig(savepath+'stackedBoois.pdf',dpi=700,bbox_inches='tight')
        plt.show()

    def computeThresholds(self):
        healthy1 = self.patientDict['Patient 1'][2:]
        healthy2 = self.patientDict['Patient 2'][2:]
        healthy3 = self.patientDict['Patient 3'][2:]
        sick1 = self.patientDict['Patient 4'][2:]
        sick2 = self.patientDict['Patient 5'][2:]
        sick3 = self.patientDict['Patient 6'][2:]

        self.healthyMean = np.mean((healthy1.mean().mean(),healthy2.mean().mean(),healthy3.mean().mean()))
        self.sickMean = np.mean((sick1.mean().mean(),sick2.mean().mean(),sick3.mean().mean()))
        self.threshold = np.mean((self.healthyMean,self.sickMean))

    def testAndPlotOnLastThree(self):
        self.computeThresholds()

        savepath = 'grafer/'
        plt.style.use('seaborn')
        fig, axs = plt.subplots(1, 4)
        row = 0
        
        for i, key in enumerate(list(self.patientDict.keys())):
            if i <= 5:
                continue 
            col=i
            dat = self.patientDict[key]

            patientMean = dat[2:].mean().mean()
            
            estimate = 'Healthy' if patientMean >= self.threshold else 'Sick'

            if i>2:
                row=0
                col-=6

            axs[col].plot(dat[0],dat[1],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[2],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[3],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[4],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[5],alpha=0.4,linewidth=0.6)
            axs[col].plot(dat[0],dat[6],alpha=0.4,linewidth=0.6)
            axs[col].set_ylim(0,50) #,200)
            axs[col].set_title(f'{key}'+f' Estimated as {estimate}',fontsize=10)
            axs[col].axhline(y=self.threshold,c='cyan',linestyle='-',linewidth=0.6)
            axs[col].axhline(y=patientMean,c='k',linestyle='-',linewidth=0.6)

            colLabel = 'Activity' if col==0 else ''
            rowLabel = 'Time in minutes' if row==1 else ''

            axs[col].set_xlabel(rowLabel,fontsize=5)
            axs[col].set_ylabel(colLabel,fontsize=5)

        fig.legend(['Tracer activity in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4',
                    'Region 5','Threshold','Mean for patient'],
                    fontsize=5)
        plt.savefig(savepath+'kunsyge.pdf',dpi=700,bbox_inches='tight')
        plt.show()

    def plotSomeDataWithMeans(self):
        savepath = 'grafer/'
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 3)
        row = 0
        self.computeThresholds

        for i, key in enumerate(list(self.patientDict.keys())):
            
            col=i
            dat = self.patientDict[key]
            mean = np.zeros(len(dat[0]))
            for j in range(len(dat[0])):
                mean[j] = (dat[2][j]+dat[3][j]+dat[4][j]+dat[5][j]+dat[6][j])/5
            
            if i>2:
                row=1
                col-=3
            axs[row,col].plot(dat[0],dat[1],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],dat[2],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],dat[3],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],dat[4],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],dat[5],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],dat[6],alpha=0.4,linewidth=0.6)
            axs[row,col].plot(dat[0],mean,alpha=0.8,linewidth=0.6)
            axs[row,col].set_ylim(0,50) #,200)
            axs[row,col].set_title(f'{key}',fontsize=10)
            axs[row,col].axhline(y=self.healthyMean,c='g',linestyle='-',linewidth=0.6)
            axs[row,col].axhline(y=self.sickMean,c='r',linestyle='-',linewidth=0.6)
            axs[row,col].axhline(y=dat[2:].mean().mean(),c='cyan',linestyle='-',linewidth=0.6)

            colLabel = 'Activity' if col==0 else ''
            rowLabel = 'Time in minutes' if row==1 else ''

            axs[row,col].set_xlabel(rowLabel,fontsize=5)
            axs[row,col].set_ylabel(colLabel,fontsize=5)

            if i >= 5:
                break
        fig.legend(['Tracer activity in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4',
                    'Region 5','Average for regions for patient',
                    'Healthy Mean','Sick Mean','Mean for patient'],
                    fontsize=5)
        plt.savefig(savepath+'allegraferMedGennemsnit.pdf',dpi=700,bbox_inches='tight')
        plt.show()

    def plotSomeData(self):
        savepath = 'grafer/'
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 3)
        row = 0
        for i, key in enumerate(list(self.patientDict.keys())):
            col=i
            dat = self.patientDict[key]
            if i>2:
                row=1
                col-=3
            axs[row,col].plot(dat[0],dat[1])
            axs[row,col].plot(dat[0],dat[2])
            axs[row,col].plot(dat[0],dat[3])
            axs[row,col].plot(dat[0],dat[4])
            axs[row,col].plot(dat[0],dat[5])
            axs[row,col].plot(dat[0],dat[6])
            axs[row,col].set_title(f'{key}',fontsize=10)
            if col==0:
                colLabel = 'Activity'
            else:
                colLabel=''
            if row==1:
                rowLabel = 'Time in minutes'
            else:
                rowLabel=''
            axs[row,col].set_xlabel(rowLabel,fontsize=5)
            axs[row,col].set_ylabel(colLabel,fontsize=5)

            if i >= 5:
                break
        fig.legend(['Tracer activity\n in arterial blood',
                    'Region 1','Region 2','Region 3','Region 4','Region 5'],
                    fontsize=5)
        plt.savefig(savepath+'allegrafer.pdf',dpi=700,bbox_inches='tight')
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