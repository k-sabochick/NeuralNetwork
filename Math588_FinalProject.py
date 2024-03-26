# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:45:29 2022

@author: katri
"""

from numpy import *
import csv
from scipy.optimize import fmin_bfgs,minimize
from random import shuffle

#data
class1start=0
class2start=121
#class3start=130


#global parameters
regParam=0.03
nSPerClass=100
nS=2*nSPerClass
n = 1 #n is the number of outputs desired

#LReLU - optional arguments to test different betas and epsilons
def lrelu(x,b=1,epsilon=0.01):
    y=[]
    for i in x:
        if i<0:
            y.append(epsilon*i)
        else:
            y.append(b*i)
    return array(y)

#Derivative
def lrelu_prime(x,b=1,epsilon=0.01):
    y=[]
    for i in x:
        if i<0:
            y.append(epsilon)
        else:
            y.append(b)
    return array(y)

#Defining our feed forward function
def feed_forward(x,W):
    W0=array(W[:k*m]).reshape((k,m))
    b0=W[k*m:k*m+k]
    W1=array(W[(m+1)*k:(m+1)*k+n*k]).reshape((n,k))
    b1=W[(m+1)*k+n*k:]
    z1=dot(W0,x)+b0
    z2=dot(W1,a1(z1))+b1
    return a2(z2)

#This block mostly comes from Dr. Cooper's code, with some adjustments from Paula and I
#accepts a column vector argument, returns mean and l2 norm of vector
def normalize(col): 
    l2norm = sqrt(sum((col-mean(col))*(col-mean(col)))/len(col))
    return mean(col),l2norm

#accepts no arguments, returns array of mean values for data & array of norm values
def normalize_cols(inMat):
    means=[]
    norms=[]
    for i in range(m):
        mn,nrm=normalize(inMat[:,i])
        means.append(mn)
        norms.append(nrm)
    meanValues=array(means)
    normValues=array(norms)
    return meanValues,normValues

#accepts a matrix, returns the normalized matrix
def normalizeByTrain(inMat):
    meanVals,normVals = normalize_cols(inMat)
    normedMat=inMat*0.0
    for k in range(len(inMat[:,0])):
        normedMat[k]=(inMat[k]-meanVals)/normVals
    return normedMat

#accepts input matrices X and Y and vector W
def ensemble_cost(W,X,Y):
    C = 0.0
    for i in range(len(X[0,:])):
        x=X[i,:]
        y=Y[i]
        C += 0.5*(dot(feed_forward(x,W)-y,feed_forward(x,W)-y)+regParam*0.5*linalg.norm(W)**2)
    return C

#Dr. Cooper's logistic function & it's derivative below + his other functions b/c I think they'll work more reliably
def S(X):
    X = X*(abs(X)<10)+10.*(X>=10)-10.*(X<=-10)
    return 1./(1+exp(-X))

def SPrime(X):
    X = X*(abs(X)<10)+10.*(X>=10)-10.*(X<=-10)
    ex = exp(-X)
    return ex/((1.+ex)*(1.+ex))

def relu(x,epsilon=0.0):
    return x*(x>0)+epsilon*x*(x<=0)

def reluPrime(x,epsilon=0.0):
    return (x>0)+epsilon*(x<=0)

def selu(x,lamda=1,alpha=1):
    return lamda*( x*(x>0)+alpha*(exp(x)-1)*(x<=0))

def seluPrime(x,lamda=1,alpha=1):
    return lamda*((x>0)+alpha*exp(x)*(x<=0))

#accepts input vector x, output vector out, and big W vector, returns a (regularized) gradient for one row
def gradient(x,out,W):#<----borrowed
    W0 = array(W[:nW0]).reshape((k,m))
    W1 = array(W[nW0+k:-n]).reshape((n,k))
    grad = W*0.0
    # Get a(z_2)
    z1 = dot(W0,x)+W[nW0:nW0+k]
    z2 = dot(W1,a1(z1))+W[-n:]
    forward = a2(z2)
    err = forward-out
    # gradient for b_1 bias weights
    finalLayerDeriv = a2p(z2)*err
    grad[-n:] = finalLayerDeriv+0.
    # gradient for W_1 weights
    grad[(nW0+k):-n] = outer(finalLayerDeriv,a1(z1)).flatten()
    # gradient for b_0 bias weights
    firstLayerDeriv = finalLayerDeriv.dot(W1)*a1p(z1)
    grad[nW0:(nW0+k)] = firstLayerDeriv+0.
    # gradient for W_0 weights
    grad[:nW0] = outer(firstLayerDeriv,x).flatten()
    return grad+regParam*W

#accepts vector W, input matrix X and output matrix Y
def gradSum(W,X,Y):
    grad = zeros(nVar)
    for i in range(len(X[:,0])): #X[:,0] is a slice of all the rows in the first column
        grad += gradient(X[i,:],Y[i],W)
    return grad

# Block for reading in the data - taken from Dr. Cooper's code#
file = csv.reader(open("parkinsons.csv","r"))
#rowIndex = 1
inData = []
outData = [] #will contain truth values for each of the three classes, i.e. 1 == [true false false], used for one-hot?
outSData = [] #will contain the numerical values corresponding to each class
testBin = 1*[0]

rows=[]
for row in file:
    rows.append(row)
    
shuffle(rows)

for row in rows:
    for ri in range(len(row)):
        row[ri] = float(row[ri])
    inData.append(row[1:])
    #if row[0]==1:
    #    testBin=[1,0]
    #elif row[0]==2:
    #    testBin=[0,1]
    #else:
    #    testBin=[1,1]
    #for i in range(1,4):
    #   testBin[i-1] = (row[0]==float(i))
    outData.append(row[0])
    outSData.append(row[0])
    
#Peeling off the training data into vectors
trainX = array(inData[:nSPerClass]+inData[class2start:class2start+nSPerClass]) #rows are input data
trainY = array(outData[:nSPerClass]+outData[class2start:class2start+nSPerClass]).astype(float) #rows are output data
trainSY = array(outSData[:nSPerClass]+outSData[class2start:class2start+nSPerClass]).astype(float) #thing to check results with
m = len(trainX[0]) #m is the number of inputs, i.e. number of columns in first row

k = int(2*m/3+n) #k is the number of hidden layers
W = random.rand((m + 1)*k+(k + 1)*n) #first estimate of the values

print('Number of samples: {}'.format(len(inData)))
print('Number of data vectors: {}'.format(nS))
print('Number of data items: {}'.format(m))
print('Number of hidden cells: {}'.format(k))
print(trainY)
nW0 = m*k
nW1 = k*n
nVar = nW0+nW1+k+n
#normalize the vector
trainX = normalizeByTrain(trainX)
#compute cost, choosing a1 to be relu and a2 to be logistic
a1 = lambda x: lrelu(x)
a1p = lambda x: lrelu_prime(x)
a2 = S
a2p = SPrime
#call training function, I'm going to use bfgs_min to solve just because it seems best
newW = W
for i in range(3):
    newW = fmin_bfgs(ensemble_cost,newW,fprime=gradSum,args=(trainX,trainY))


#the test data
testX = array(inData[nSPerClass:class2start]+inData[class2start+nSPerClass:]) #rows are the rest of the input data
testY = array(outData[nSPerClass:class2start]+outData[class2start+nSPerClass:]).astype(float) #rows are rest of output data
testSY = array(outSData[nSPerClass:class2start]+outSData[class2start+nSPerClass:]).astype(float) #thing to check results with    

testX = normalizeByTrain(testX)

predtrain=[]
for i in arange(len(trainX[:,0])):
    x=trainX[i,:]
    result=feed_forward(x,newW)
    predtrain.append(result)
    
realpredtrain=(array(predtrain)+0.5).astype(int)

#now feed forward again but on test data with theoretically more accurate W
pred=[]
for i in arange(len(testX[:,0])): #i iterates over number of rows in training data
    x=testX[i,:] #x grabs the i-th row
    result=feed_forward(x,newW)
    pred.append(result)

realpred=(array(pred)+0.5).astype(int)
print(array(pred))
print(realpred)

right=0
wrong=0
for i in range(len(trainY)):
    if trainY[i]==realpredtrain[i]:
        right=right+1
    else:
        wrong=wrong+1
print('Percentage correct for train:',(right/(right+wrong))*100)      


right=0
wrong=0
for i in range(len(testY)):
    if testY[i]==realpred[i]:
        right=right+1
    else:
        wrong=wrong+1
print('Percentage correct for test:',(right/(right+wrong))*100)
