import numpy as np
import random
np.random.seed(0)
import matplotlib.pyplot as plt
import pandas as pd
import time
import random 
from operator import add
from sklearn.neighbors import NearestNeighbors as NN
import math
import decimal
from scipy.signal import savgol_filter
from scipy.stats import levene
from scipy.sparse import csc_matrix, linalg as sla, csr_matrix,hstack,vstack


def partition(lst, n):
    #partition a python list into n sublists. keeps order of list
#    random.shuffle(lst)
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

############################################################################


def getNonUniformTimeSliceBins(T,tMin,tMax,nSlices):
    # construct the time slices themselves
    totData=np.sum(np.array([len(x) for x in T]))
    sSlices=int(np.floor(totData/nSlices))
    fT=[]
    for i in range(len(T)):
        tB=T[i]
        for j in range(len(tB)):
            t=tB[j]
            fT.append([t,i,j])
#    print(len(fT))    
    fT.sort()
    binsRough=partition(fT, nSlices)
    sliceLen=[]
    Tnew=[[] for x in range(len(T))]
    for k in range(len(binsRough)):
        if k==0:
            dt=binsRough[k][len(binsRough[k])-1][0]-binsRough[k][0][0]
        else:
            dt=binsRough[k][len(binsRough[k])-1][0]-binsRough[k-1][len(binsRough[k-1])-1][0]
        sliceLen.append(dt)
#        binsRough[k].sort(key=Second)
        for i in range(len(binsRough[k])):
            dataLabel=binsRough[k][i][1]
            Tnew[dataLabel].extend([k+1])
#  return binsRough, fT
    return Tnew, sliceLen

############################################################################

def getRawTimeSlice(xClassIn, binIn, nSlices):
  # assign the data to time slices
    
    tS=[[] for i in range(nSlices)]
    for i in range(len(xClassIn)):
        for j in range(len(xClassIn[i])):
#      print(i,j)
              binNum=int(binIn[i][j])
              tS[binNum-1].append(xClassIn[i][j])
    return tS
    
############################################################################ 
def generateTimePoints(tSlice, nPoints, nNeighbors=3):
  #function to use SMOTE on each time slice to generate synthetic data within each class
  #nPoints is the number of synthetic data points you wish to compute. 
  #the idea here is to just generate a ton in each class and pick when needed
  #so for this, we just generate the same number for each time slice. because its easier. 
  #if sampling for imputation step is done without replacement, must choose a nPoints >= (Num. Tot. Obs)/(Num. Slices). Sample code: sum([len(x) for x in X])/nSlices
    nulls=['None', 'Null', 'NaN', 'nan', 'NAN', np.nan]
    tSliceSyn=[]
    for k in range(len(tSlice)): #loop through time points
#    print(k)
        nFeats=len(tSlice[k][0])
        tSliceInt=[]
        for i in range(nFeats):
            data=[[tSlice[k][j][i]] for j in range(len(tSlice[k])) if tSlice[k][j][i] not in nulls]
            if nNeighbors>=len(data):
                nNeighbors=len(data)-1
                runs=math.ceil(nPoints/(nNeighbors*len(data)))
                neigh=NN(n_neighbors=nNeighbors)
                neighbors=neigh.fit(data)
                tSliceSynT=[]
                m=0
                while m<= runs:
                    for x in data:
                        query=[x]
                        A=neigh.kneighbors(query, return_distance=False)
                        for n in A[0]:
                            if n != data.index(x):
                                Xn=data[n]
                                l=random.random()
                                xm=[-y for y in x]
                                t1=list(map(add, Xn, xm)) #compute l(Xn-X)
                                t1=[l*x for x in t1]
                                t=list(map(add, x, t1)) #comput X+l(Xn-X)
                                tSliceSynT.append(t) 
                    m+=1
                tSliceInt.append(tSliceSynT)


            else:
                neigh=NN(n_neighbors=nNeighbors)
                neighbors=neigh.fit(data)
                tSliceSynT=[]
                while len(tSliceSynT)<=nPoints:
                    for x in data:
                        query=[x]
                        A=neigh.kneighbors(query, return_distance=False)

                        for n in A[0]:
                            if n != data.index(x):
                                Xn=data[n]
                                l=random.random()
                                xm=[-y for y in x]
                                t1=list(map(add, Xn, xm)) #compute l(Xn-X)
                                t1=[l*x for x in t1]
                                t=list(map(add, x, t1)) #comput X+l(Xn-X)
                                tSliceSynT.append(t) 
                tSliceInt.append(tSliceSynT)      
        vecs=[[tSliceInt[p][q][0] for p in range(nFeats)] for q in range(nPoints)]
        tSliceSyn.append(vecs)
    return tSliceSyn

############################################################################    

def imputeTimeSlices(X,T,tSliceSyn,nFix):
  #function for imputing time slices into the data. 
  #T must be time bins
  #nFix denotes the number of features that are time independent. 
  #Put these first in your feature vector as such: features=[time independent features, time dependent features]
    nulls=['None', 'Null', 'NaN', 'nan', 'NAN', np.nan]
    Xout=[]
    nSlice=len(tSliceSyn)
    bad=[[] for x in range(len(tSliceSyn))]
    for i in range(len(X)):
        for j in range(len(X[i])):
            for q in range(len(X[i][j])):
                t=int(T[i][j])
                if X[i][j][q] in nulls:
                    c=random.choice(tSliceSyn[t-1])
                    cc=c[q]
                    X[i][j][q]=cc
        intSlice=[[] for x in range(len(tSliceSyn))]
        for k in range(len(X[i])):
            t=int(T[i][k])
            intSlice[t-1].append(X[i][k])
        for p in range(len(intSlice)):
            if len(intSlice[p])==0:
                c=random.choice(tSliceSyn[p])
                while c in bad[p]:
                    c=random.choice(tSliceSyn[p])
                tSliceSyn[p].append(c)
                cNew=[]
                for q in range(len(c)):
                    if q<nFix:
                        cNew.append(X[i][0][q])
                    else:
                        cNew.append(c[q])
#                    print(c[:nFix]==X[i][0][:nFix])
                intSlice[p]=cNew
            elif len(intSlice[p])>1:
                A=np.array(intSlice[p])
                intSlice[p]=np.mean(A,axis=0).tolist()
            elif len(intSlice[p])==1:
                intSlice[p]=intSlice[p][0]
        Xout.append(intSlice)
    return Xout


def removeDegeneracies(X,bins):
    #this is in place so either pass a copy of your data or just run this exactly once
    #ideally just run this after you get your data assembled correctly i.e. list of shape (samps, obs, feats)
    tBinsNew=[]
    for i in range(len(X)):
#        print(i)
        tBinSet=list(set(bins[i]))
#        print(bins[i])
        degen=[[item,idx] for idx, item in enumerate(bins[i]) if item in bins[i][:idx]]
        dupInd=[]
#        print(f'before: {len(X[i])}')
        for t in tBinSet:
            if bins[i].count(t)>1:
#                print(t)
                indicies=[x[1] for x in degen if x[0]==t]
                dupInd.extend(indicies)

                vals=[X[i][k] for k in indicies]
                vals.append(X[i][min(indicies)-1])
                newVal=list(np.array(vals).mean(axis=0))
                X[i][min(indicies)-1]=newVal
        X[i]=[X[i][k] for k in range(len(X[i])) if k not in dupInd]
        tBins=[x for x in list(set(bins[i]))]
        tBins.sort()
        tBinsNew.append(tBins)
#        print(f'after: {len(X[i])}')
#        print('')
    return X,tBinsNew

# convinient functions for list keys        
def First(val):
    return val[0]
def Second(val): 
    return val[1]
def Third(val): 
    return val[2]
def Fourth(val): 
    return val[3]



def constructFirstPart(n,samp,tSliceSyn,nFix=0,nSubSamp=100, sig=1,K=0, norm='forward'):
    # constructs ttransition matrix from first observation to first time slice. not necessary if you assign initial point seperately
    T=np.zeros((1+(n-1)*nSubSamp,1+(n-1)*nSubSamp))
    
    for m in range(n-1):
        synData=random.sample(tSliceSyn[n-m-2],nSubSamp) #sample the synthetic data
        synData=np.array(synData)
        
        if m==0:
            s=samp
            TT=np.array([s])
            #print(TT.shape)
            dist=np.array([np.linalg.norm(np.array(s)-synData[i,:])**2 for i in range(len(synData))]) 
            prob=np.exp(-dist/sig)
            prob=np.multiply(dist**K,prob)
#            prob=prob/np.sum(prob)
            prob=np.insert(prob,0,0)
            prob=np.append(prob,np.zeros((n-m-2)*nSubSamp))
            T[:,m]=prob
            TT=np.append(TT,synData, axis=0)

        elif m >0:
            TT=np.append(TT,synData, axis=0)
            for j in range(len(synOld)):
                s=synOld[j]
                dist=np.array([np.linalg.norm(np.array(s)-synData[i,:])**2 for i in range(len(synData))])
                prob=np.exp(-dist/sig)
                prob=np.multiply(dist,prob)
#                prob=prob/np.sum(prob)

                prob=np.append(np.zeros(1+m*nSubSamp),prob)
                prob=np.append(prob,np.zeros((n-m-2)*nSubSamp))

                T[:,1+j+(m-1)*nSubSamp]=prob

                #print(TT.shape)

#                        print(len(prob))
        synOld=synData
    if norm=='symmetric':
        normR=np.sqrt(T.sum(axis=1))
        normC=np.sqrt(T.sum(axis=0))
        normR[normR==0]=1
        normC[normC==0]=1
        normR=np.reciprocal(normR)
        normC=np.reciprocal(normC)
        T=T*normR[:,None]
        T=T*normC
        normC=T.sum(axis=0)
        normC[normC==0]=1
        normC=np.reciprocal(normC)
        T=T*normC
        
    elif norm=='forward':
        normC=T.sum(axis=0)
        normC[normC==0]=1
        normC=np.reciprocal(normC)
        T=T*normC
    elif norm=='backward':
        normR=T.sum(axis=1)
        normR[normR==0]=1
        normR=np.reciprocal(normR)
        T=T*normR[:,None]
    if nFix>0: #fix the fixed features
#        print('working!')
        for i in range(nFix):
            TT[:,i]=np.full_like(TT[:,i],samp[i])
    return csc_matrix(T),csc_matrix(TT)



def constructRest(nStart,nStop,samp,tSliceSyn,nFix=0,nSubSamp=100,sig=1,K=0,norm='symmetric'):
    #construct the transition matricies for intermediate steps
    
#    print(nStart,nStop)
    T=np.zeros((1+(nStop-nStart-1)*nSubSamp,1+(nStop-nStart-1)*nSubSamp))
    TT=np.array([samp])

    for m in range(nStart,nStop-1):
        M=m-nStart
        synData=random.sample(tSliceSyn[m],nSubSamp) #sample the synthetic data
        synData=np.array(synData)
        if m==nStart:
            dist=np.array([np.linalg.norm(np.array(samp)-synData[i,:])**2 for i in range(len(synData))]) 
            dist=(dist-dist.min())/(dist.max()-dist.min())
            prob=np.exp(-dist/sig)
            prob=np.multiply(dist**K,prob)
#            prob=prob/np.sum(prob)
            prob=np.insert(prob,0,0)
            prob=np.append(prob,np.zeros((nStop-nStart-M-2)*nSubSamp))
            T[:,M]=prob
#            print(m,TT.shape,synData.shape)
            TT=np.append(TT,synData, axis=0)
#            print(m,TT.shape)

        elif m >nStart:
            TT=np.append(TT,synData, axis=0)
#            print(m,TT.shape)
            for j in range(len(synOld)):
                s=synOld[j]
                dist=np.array([np.linalg.norm(np.array(s)-synData[i,:])**2 for i in range(len(synData))])
                dist=(dist-dist.min())/(dist.max()-dist.min())
                prob=np.exp(-dist/sig)
#                print(prob)
#                prob=prob/np.sum(prob)
                prob=np.multiply(dist**K,prob)

                prob=np.append(np.zeros(1+M*nSubSamp),prob)
                prob=np.append(prob,np.zeros((nStop-nStart-M-2)*nSubSamp))

                T[:,1+j+(M-1)*nSubSamp]=prob

        synOld=synData
    if norm=='symmetric':
        normR=np.sqrt(T.sum(axis=1))
        normC=np.sqrt(T.sum(axis=0))
        normR[normR==0]=1
        normC[normC==0]=1
        normR=np.reciprocal(normR)
        normC=np.reciprocal(normC)
        T=T*normR[:,None]
        T=T*normC
        normC=T.sum(axis=0)
        normC[normC==0]=1
        normC=np.reciprocal(normC)
        T=T*normC
    elif norm=='forward':
        normC=T.sum(axis=0)
        normC[normC==0]=1
#        print(normC)
        T=T*np.reciprocal(normC)
    elif norm=='backward':
        normR=T.sum(axis=1)
        normR[normR==0]=1
        normR=np.reciprocal(normR)
        T=T*normR[:,None]
    if nFix>0: #fix the fixed features
#        print('working!')
        for i in range(nFix):
            TT[:,i]=np.full_like(TT[:,i],samp[i])
            
    
    return csc_matrix(T),csc_matrix(TT)



def constructTransitionMatrix(samp,tSamp,tSliceSyn,nFix=0,nSubSamp=100,sig=1,K=0,norm='symmetric'):
    #this is to construct the full transition matricix for a single sample, outputs a dict with transition matricies and samples
    
#    need=[x for x in range(len(tBins)) if x not in tSamp]  
    out={}
    for k in range(len(tSamp)):
#        print(tSamp[k])
        if k==0:
            n=tSamp[k]
            if n!=1:
                T,TT = constructFirstPart(n,samp[k],tSliceSyn,nFix,nSubSamp,sig,K,norm)
                out[f'{n}-{1}']=[T,TT]
                
                if tSamp[k]==tSamp[k+1]-1:
#                    print('woop')
                    continue
                else:
                    nStart=tSamp[k]
                    nStop=tSamp[k+1]
                    T,TT=constructRest(nStart,nStop,samp[k],tSliceSyn,nFix,nSubSamp, sig,K,norm)
#                    print(T.max())
#                print('Ok!')
                    out[f'{nStart}-{nStop}']=[T,TT] 
#                print((TT.toarray(),TT.shape))
            elif (n==1):
                if (tSamp[k+1]==2):
                    continue
#                print('1')
                nStart=tSamp[k]
                nStop=tSamp[k+1]
                T,TT=constructRest(nStart,nStop,samp[k],tSliceSyn,nFix,nSubSamp, sig,K,norm)
#                print('Ok!')
                out[f'{nStart}-{nStop}']=[T,TT] 
#                print(TT.toarray())
        elif k==len(tSamp)-1:
#            nStart=tSamp[k-1]
#            nStop=tSamp[k]
#            T,TT=constructRest(nStart,nStop,samp[k],tSliceSyn,nFix,nSubSamp, sig,K,norm)
#            out[f'{nStart}-{nStop}']=[T,TT] 
            if tSamp[k]==len(tSliceSyn):
                continue
            else:
                nStart=tSamp[k]
                nStop=len(tSliceSyn)+1
#            print(nStart-nStop)
                T2,TT2 = constructRest(nStart,nStop,samp[k],tSliceSyn,nFix,nSubSamp, sig,K,norm)
                out[f'{nStart}-{nStop-1}']=[T2,TT2]
        else:
            if tSamp[k]==tSamp[k+1]-1:
#                print(f'why!:{tSamp[k], tSamp[k+1]}')
                continue
            else: 
                nStart=tSamp[k]
                nStop=tSamp[k+1]
                T,TT=constructRest(nStart,nStop,samp[k],tSliceSyn,nFix,nSubSamp, sig,K,norm)
                out[f'{nStart}-{nStop}']=[T,TT]     
        
    return out


def getTrajectoryLocal(samp,tSamp,tSliceSyn,nIt=25,nFix=0,nSubSamp=100,sig=1,K=0, norm='symmetric'):
# here we get the imputed trajectories by maximizing marginals,  averaging over nIt iterartions    
    valsOut=np.empty((nIt,len(tSliceSyn),len(samp[0])))
    for q in range(nIt):
        valsInt=np.empty((len(tSliceSyn),len(samp[0])))
        tOut=constructTransitionMatrix(samp,tSamp,tSliceSyn,nFix,nSubSamp,sig,K,norm)
        for t in tOut:
            if (list(tOut).index(t)==0)&(tSamp[0]!=1):
                start=int(t.split('-')[0])
                stop=int(t.split('-')[1])
            else: 
                start=int(t.split('-')[0])
                stop=int(t.split('-')[1])
#            print((start,stop))
            tMat=tOut[t][0]
            tVals=tOut[t][1]
            vals=[]
            p=np.zeros(tMat.shape[0])
            p[0]=1
            p=csc_matrix(p)
            p=p.transpose()
            if (stop==20)|(stop==1):
                s=0
            else: 
                s=-1
#            print((stop,s))
            for qq in range(abs(start-stop)+s):
                p=tMat.dot(p)
                if (list(tOut).index(t)==0)&(tSamp[0]!=1):
                    vals.insert(0,tVals[np.argmax(p)])
#                    print(tVals.toarray()[np.argmax(p)])
                    d=-1
                else: 
                    vals.append(tVals[np.argmax(p)])
                    d=0
            if stop==1:
                for i in range(len(vals)):
#                    print((i,d,stop))
#                print((vals[i].toarray()[0],start+i+d))
                    valsInt[stop+i+d,:]=vals[i].toarray()[0]
#                    print(f'valsInt {stop+i+d}: {valsInt[stop+i+d,:]}')
            else:                     
                for i in range(len(vals)):
#                    print((i,d,start))
#                print((vals[i].toarray()[0],start+i+d))
                    valsInt[start+i+d,:]=vals[i].toarray()[0]
#                    print(f'valsInt {start+i+d}: {valsInt[start+i+d,:]}')
            for i in range(len(tSamp)):
                j=tSamp[i]
                valsInt[j-1,:]=samp[i]
#                print(f'valsInt {j-1}: {valsInt[j-1,:]}')
#            print(f'valsInt: {valsInt}')
        valsOut[q,:,:]=valsInt
    return valsOut
            

def getTrajectoryGlobal(samp,tSamp,tSliceSyn,nIt=25,nFix=0,nSubSamp=25,sig=1,K=0,norm='forward'):
    # gives fully imputed sample using max-product algorithm for imputation, averaging over nIt iterartions 
    
    valsOut=np.zeros(shape=(nIt,len(tSliceSyn),len(samp[0])))
    for Q in range(nIt):
        tOut=constructTransitionMatrix(samp,tSamp,tSliceSyn,nFix,nSubSamp,sig,K,norm)
        vals=np.zeros(shape=(len(tSliceSyn),len(samp[0])))
        for t in tOut:
            posOut=[]
            start=int(t.split('-')[0])
            stop=int(t.split('-')[1])
#            print(t)
            T=tOut[t][0].toarray()
            TT=tOut[t][1].toarray()
            if (stop in [1,len(tSliceSyn)])&(stop not in tSamp):
                sampStop=None
                prob=np.ones(shape=TT.shape[0])
            else:
                sampStop=samp[tSamp.index(stop)]
                dist=np.array([np.linalg.norm(np.array(sampStop)-TT[j])**2 for j in range(TT.shape[0])])
                dist=(dist-dist.min())/(dist.max()-dist.min())
                prob=np.exp(-dist/sig)
                prob=prob/np.sum(prob)

            if (abs(stop-start)==1)&(stop in [1,len(tSliceSyn)]):
                if stop==1:
                    if (norm=='forward')|(norm=='symmetric'):
                        A=T[:,0]
                        vals[0,:]=TT[A.argmax()]
                    elif norm=='backward':
                        dist=np.array([np.linalg.norm(samp[0]-TT[j])**2 for j in range(TT.shape[0])])
                        dist=(dist-dist.min())/(dist.max()-dist.min())
                        probInt=np.exp(-dist/sig)
                        probInt=probInt/np.sum(probInt)
                        A=T[:,0]*probInt
                        vals[0,:]=TT[A.argmax()]
                        
                elif stop==len(tSliceSyn):
                    #prob=np.zeros(shape=nSubSamp+1)
                    if (norm=='forward')|(norm=='symmetric'):
                    #    for j in range(nSubSamp):
                    #        dist=np.array([np.linalg.norm(TT[j+1]-x)**2 for x in [samp[-1]]+random.sample(tSliceSyn[-1], nSubSamp)])
                    #        probInt=np.exp(-dist/sig)
                    #        probInt=probInt/np.sum(probInt)
                    #        prob[j+1]=probInt[0]
                        A=T[:,0]
                        vals[-1,:]=TT[A.argmax()]
                    elif norm=='backward':
                        dist=np.array([np.linalg.norm(samp[-1]-TT[j])**2 for j in range(TT.shape[0])])
                        dist=(dist-dist.min())/(dist.max()-dist.min())
                        probInt=np.exp(-dist/sig)
                        probInt=probInt/np.sum(probInt)
                        A=T[:,0]*probInt
                        vals[-1,:]=TT[A.argmax()]



            elif abs(stop-start)==2:
                P=1
                A=T[:,0]
                A=A*prob
                np.nan_to_num(A,copy=False)
                vals[stop-2,:]=TT[T[:,0].argmax()]

            elif abs(stop-start)==3:
                A=T[:,0]
                A=T*A
                np.nan_to_num(A,copy=False)
                maxMess=A.max(axis=0)
                maxInd=A.argmax(axis=0)
                messInt=np.zeros(shape=nSubSamp)
                for i in range(len(maxInd)):
                    maxMess[i]=maxMess[i]*prob[maxInd[i]] 

    #            print(maxMess)
                vals[stop-2,:]=TT[maxMess.argmax()]
                vals[stop-3,:]=TT[maxInd[maxMess.argmax()]]

            else:
                if (stop in [1,len(tSliceSyn)])&(stop not in tSamp):
                    rang=abs(start-stop)-1
                    print('yes')
                else:
                    rang=abs(start-stop)-2
                A=T[:,0]
                A=T*A
#                print(T)
                np.nan_to_num(A,copy=False)
                maxMess=A.max(axis=1)
                maxInd=A.argmax(axis=1)
                ind=np.where(maxInd>0)[0]
#                print(ind)
                maxMess=maxMess[maxMess>0]
                maxInd=maxInd[maxInd>0]
                mess=np.empty(shape=(rang,nSubSamp,3))
#                print(maxMess, maxInd)
#                print(A)
                for i in range(maxMess.shape[0]):
                    x=maxMess[i]
                    y=maxInd[i]
                    z=ind[i]
                    mess[0,i,0]=y
                    mess[0,i,1]=z
                    mess[0,i,2]=x

                for k in range(1,rang):
                    I=mess[k-1,:,1]
                    M=mess[k-1,:,2]
                    A=np.zeros(shape=TT.shape[0])
                    n1=int(I[0])
                    n2=int(I[-1])+1
#                    print('')
#                    print(k)
#                    print('')
#                    print(n1,n2)
#                    print(M)
                    A[n1:n2]=M
                    A=T*A
                    maxMess=A.max(axis=1)
                    maxInd=A.argmax(axis=1)
                    ind=np.where(maxInd>0)[0]
                    maxMess=maxMess[maxMess>0]
                    maxInd=maxInd[maxInd>0]
                    for j in range(maxMess.shape[0]):
                        x=maxMess[j]
                        y=maxInd[j]
                        z=ind[j]
                        mess[k,j,0]=y
                        mess[k,j,1]=z
                        mess[k,j,2]=x
                        
                kk=mess.shape[0]-1
                ii=mess[kk,:,2].argmax()
                pos=mess[kk,ii,1]
                posPrev=mess[kk,ii,0]
                if stop==1:
                    inc=0
                    val1=TT[int(posPrev)]
                    val=TT[int(pos)]
                else: 
                    inc=start+kk
                    val=TT[int(posPrev)]
                    val1=TT[int(pos)]
                vals[inc,:]=val
                vals[inc+1,:]=val1
#                print(mess)
                for k in range(1, mess.shape[0]):
#                    print(posPrev)
                    kk=mess.shape[0]-k-1
                    messInt=mess[kk,mess[kk,:,1]==posPrev,:]
                    ii=messInt[:,2].argmax()
                    posPrev=messInt[ii,0]
                    val=TT[int(posPrev)]
                    if stop==1:
                        inc=k+1
                    else: 
                        inc=start+kk
                    vals[inc,:]=val
#                    print(posPrev)
                
                    
            for i in range(len(tSamp)):
                ind=tSamp[i]
                vals[ind-1,:]=samp[i]

        valsOut[Q,:,:]=vals
    return valsOut


def getTrajectory(samp,tSamp,tSliceSyn,nIt=25,nFix=0,nSubSamp=100,sig=1,K=0,mode='global',norm='symmetric'):
    # obtain the full trajectory for each sample, imputing first and last slice and then computing the trajectories
    if 1 not in tSamp:
#        print('adding 1')
        tSamp.insert(0,1)
        samp.insert(0,random.choice(tSliceSyn[0]))
        if nFix!=0:
            samp[0][:nFix]=samp[1][:nFix]
    if len(tSliceSyn) not in tSamp:
#        print(f'adding {len(tSliceSyn)}')
        tSamp.append(len(tSliceSyn))
        samp.append(random.choice(tSliceSyn[-1]))
        if nFix!=0:
            samp[-1][:nFix]=samp[1][:nFix]

    if mode=='global':
        return getTrajectoryGlobal(samp,tSamp,tSliceSyn,nIt,nFix,nSubSamp,sig,K,norm)
    if mode=='local':
        return getTrajectoryLocal(samp,tSamp,tSliceSyn,nIt,nFix,nSubSamp,sig,K,norm)

def imputeMixed(data,tBins,tSliceSyn,nf=25,nb=25,nFix=0,nSubSamp=100,sig=1,K=0, mode='global',verbose=False):
    # imputes based on a mixture of forward and backward processes. reccomended for use
    valsOut=np.zeros(shape=(len(data),len(tSliceSyn), len(data[0][0])))
    for i in range(len(data)):
        start=time.time()
        samp=data[i]
        tSamp=tBins[i]
        valsF=getTrajectory(samp,tSamp,tSliceSyn,nf,nFix,nSubSamp,sig,K,mode,norm='forward')
        valsB=getTrajectory(samp,tSamp,tSliceSyn,nb,nFix,nSubSamp,sig,K,mode,norm='backward')
        vals=np.concatenate((valsF,valsB)).mean(axis=0)
        valsOut[i,:,:]=vals
        end=time.time()
        if verbose==True:
            print(f'sample {i} took {end-start} s')
    return valsOut


def imputeMeanTimeSlices(data,tPoints,tSliceSyn):
    # impute based on the mean of each slice
    dataNew=np.zeros(shape=(len(data),len(tSliceSyn), len(data[0][0])))
    for i in range(len(data)):
        bins=tPoints[i]
        samp=np.array(data[i])
    for j in range(len(tSliceSyn)):
        if j+1 in bins:
            I=bins.index(j+1)
            dataNew[i,j,:]=samp[I]
        elif j+1 not in bins:
            dataNew[i,j,:]=np.array(tSliceSyn[j]).mean(axis=0)
    return dataNew.tolist()


def imputeMedianTimeSlices(data,tPoints,tSliceSyn):
    # impute based on the median of each slice
    dataNew=np.zeros(shape=(len(data),len(tSliceSyn), len(data[0][0])))
    for i in range(len(data)):
        bins=tPoints[i]
        samp=np.array(data[i])
    for j in range(len(tSliceSyn)):
        if j+1 in bins:
            I=bins.index(j+1)
            dataNew[i,j,:]=samp[I]
        elif j+1 not in bins:
        dataNew[i,j,:]=np.median(np.array(tSliceSyn[j]),axis=0)
    return dataNew.tolist()
        
                   


def timesToBins(times, inTime, outTime):
  #gives bins corresponding to different time points
    outT=len([t for t in times if t<outTime])
    inT=len([t for t in times if t<inTime])
    return inT, outT

def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed
