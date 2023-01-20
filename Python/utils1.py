import numpy as np
import tensorflow as tf
import scipy.io
from random import shuffle
import copy
import scipy.ndimage as sn
import random
import math


def heaviside(phi):
    epsilon = 0.001
    H = 0.5*tf.abs(1+(2.0/math.pi)*tf.atan(phi/epsilon) )
    return H



def ReadData(fileName):
    Dict = scipy.io.loadmat(fileName)
    return Dict

def MyGrad(U, X):
    gradu = tf.gradients(U, X)
    u_x = gradu[0][:, 0]
    u_x = tf.squeeze(u_x)
    u_x = tf.reshape(u_x, [-1, 1])
    u_y = gradu[0][:, 1]
    u_y = tf.squeeze(u_y)
    u_y = tf.reshape(u_y, [-1, 1])
    return u_x,u_y




def CalcValuesCompact(Xpool,Dict,keys):


    tempVar = np.zeros((Xpool.shape[0], 1), dtype=np.float32)
    tempMat = np.zeros_like(Dict['Omega'],dtype=np.float32)
    aVec =[]
    AVec = []
    D = {}
    for i in range(len(keys)):
        a = copy.deepcopy(tempVar)
        aVec.append(a)
        curKey = keys[i]
        if curKey in Dict.keys():
            AVec.append(Dict[curKey])
        else:
            AVec.append(tempMat)


    for i in range(Xpool.shape[0]):
        y = Xpool[i, 1] - 1
        x = Xpool[i, 0] - 1

        for k in range(len(keys)):
            A = AVec[k]
            aVec[k][i] = A[y,x]
            #curKey = keys[k]
            #if curKey in Dict.keys():
            #    A = Dict[curKey]
            #    aVec[k][i] = A[y, x]
            #else:
            #    aVec[k][i]=0




    for k in range(len(keys)):
        a = aVec[k]
        a = np.squeeze(a)
        a = np.reshape(a, [Xpool.shape[0], 1])
        D.update({keys[k]: a})




        
    return D
        
        


def ExtractX(Dict):
    YOMEG = Dict['YOmeg']
    XOMEG = Dict['XOmeg']
    Xall = np.zeros((XOMEG.shape[0], 2), dtype=np.int64)
    Xall[:, 0] = np.transpose(XOMEG)
    Xall[:, 1] = np.transpose(YOMEG)

    XallRand = ShuffleX(Xall)


    return Xall,XallRand


def ShuffleX(Xt):

    Num = Xt.shape[0]
    list = np.linspace(0, Num - 1, Num)
    random.seed(4)
    random.shuffle(list)
    XRand = np.zeros_like(Xt)
    for i in range(Num):
        idx = int(list[i])
        XRand[i, 0] = Xt[idx,0]
        XRand[i, 1] = Xt[idx,1]
    return XRand

def GetImageandGrads(U,Dict,key):

    Ucopy = copy.deepcopy(np.float64(U))
    W = Dict['W']
    Omega = Dict['Omega']

    se  =np.ones((5,5))
    Wdil = sn.binary_dilation(W,structure=se)
    Unan = (Ucopy)
    Unan[Omega == 0] = np.nan
    uytemp, uxtemp = np.gradient(Unan)
    gx = uxtemp

    gx[Wdil==True]=0
    gx[np.isnan(gx)] = 0

    gy = uytemp
    gy[Wdil == True] = 0
    gy[np.isnan(gy)] = 0

    if key == 'U1':
        gx1 = CorrectImageSize(gx, Omega)
        gy1 = CorrectImageSize(gy, Omega)
    else:
        gx1 = gx
        gy1 = gy

    return gx1,gy1


def CorrectImageSize(U,Omega):

    sc = int(round(U.shape[0]/2))
    mn = np.min(U)
    epsil = 0.1

    uxt = copy.deepcopy(U)
    mask = np.ones_like(U, dtype=bool)
    mask[U == 0 | np.isnan(U)] = False
    uxt[mask == True] = U[mask == True] + mn + epsil

    ux1=sn.morphology.grey_dilation(uxt,footprint=np.ones((2,2)))
    offset = ux1[sc, sc] - uxt[sc, sc]
    #print(offset)
    ux1[ux1 > 0] = ux1[ux1 > 0] - mn - epsil - offset



    ux1[mask == True] = U[mask == True]
    ux1[Omega==0]=0


    return ux1


