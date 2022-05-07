import scipy.integrate as spi
import pandas as pd 
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
plt.style.use("seaborn")

path = "Data/"
fileList = sorted(glob.glob(os.path.join(path,"*.csv")))
hold = fileList.pop(1) #maintain order
fileList.append(hold) 

data = np.empty((len(fileList),158,7))

for i, filename in enumerate(fileList):
    df = pd.read_csv(str(filename),header=None)
    data[i] = np.array(df)

data[:,0,:] = 0 #one solution to zero-row-issue


def system_matrix(data,tissue_num):
    r = tissue_num + 1
    A = np.zeros((data.shape[0],4))
    col1 = []
    col2 = [] 
    col3 = []
    col4 = []
    for i in range(A.shape[0]):
        
        if i: #>0first row is zero-measurement (only in some cases!)
            col1.append(spi.simpson(data[:i,1],x=data[:i,0],even='avg'))
            col2.append(spi.simpson(col1[:i],x=data[:i,0],even='avg'))
            col3.append(spi.simpson(data[:i,2],x=data[:i,0],even='avg'))
            col4.append(spi.simpson(col3[:i],x=data[:i,0],even='avg'))
                
    return col1,col2,col3,col4


def leastSquare(A,data,tissue_num):
    r = tissue_num + 1
    B = np.array([A]).squeeze().T
    #using pseudoinverse (old implementation, works but unstable)
    pinv = np.linalg.pinv(B) #get pseudoinverse
    alpha = pinv.dot(data[1:,r]) #least-squares fit 
    
    #using qr-decomposition
    q,R = np.linalg.qr(B)
    b = data[1:,r] #drop 0th element for having same dimensionality in integrals and goal
    p = np.dot(q.T,b)
    alpha = np.linalg.solve(R,p)#np.dot(np.linalg.inv(R),p)
    print(alpha)
    
    #using normal least squares, with new conditional 
    #alpha = np.linalg.lstsq(B,data[1:,r],rcond='None')[0]
    
    #b = data[:-1,r]
    #global alphaG
    #alphaG = lsq_linear(B,b)#,bounds=(0,np.inf))
    #alpha = nnls(B,b)
    
    return alpha #alphaG.x#alpha[0][-1].x

def get_coefficient_array(patient,data):
    patient1 = data[patient-1] #cut to slice corresponding requested patient-number
    alphas = []
    for i in range(5):
        A = system_matrix(patient1,i+1) #get system-matrix
        #print condition-number
        G = np.array(A)
        print("condition number patient {}".format(i+1)+str(np.linalg.cond(G.T)))
        vals = leastSquare(A,patient1,i+1)
        alphas.append(vals)
    
    p_matrix = alphas
    return p_matrix

def get_ks(p_means):
    p1 = p_means[0]
    p2 = p_means[1]
    p3 = p_means[2]
    p4 = p_means[3]
    k1 = p1
    k2 = -(p1*p3 + p2)/p1
    k3 = -(p1**2*p4 - p1*p2*p3 - p2**2)/((p1*p3 + p2)*p1)
    k4 = p4*p1/(p1*p3 + p2)
    
    return k1,k2,k3,k4

'''p_vals_est = get_coefficient_array(1,data)
p_vals_mean = p_vals_est.mean(1).squeeze()
k_est = get_ks(p_vals_mean)
'''
k_est = np.zeros([10,5,4])
p_vals_est = np.zeros([10,5,4])
for i in range(data.shape[0]):
    p_vals_est[i] = get_coefficient_array(i+1,data)
    #p_vals_mean = p_vals_est[i].mean(1).squeeze()
    for j in range(5):    
        k_est[i,j,:]= (get_ks(p_vals_est[i,j]))
        k_means = np.mean(k_est,axis=1)

                    


health_means = k_means[:6][:].mean(axis=0)
sick_means = k_means[6:][:].mean(axis=0)