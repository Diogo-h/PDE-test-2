
import numpy as np
import tensorflow as tf
import os
import copy
import time
import scipy.ndimage as sn

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from matplotlib import pyplot as plt
from BuildNet import *
from utils1 import *
import scipy.misc
import json


def calcError(f_gt, f, Omega):
    e = np.multiply(f - f_gt, Omega)
    err = np.mean(np.square(e))
    return err


def SetX(Xall,inputSize,CorrectB,CorrectS):

    Xnor = np.zeros([Xall.shape[0], inputSize])

    Xnor[:,0] = (Xall[:,0]-CorrectB)/CorrectS
    Xnor[:,1] = (Xall[:,1]-CorrectB)/CorrectS
    return Xnor


def GetImages(sess,X,Xall,Dict,dictList,key,u,inputSize,CorrectB,CorrectS):


    Xnor = SetX(Xall, inputSize, CorrectB, CorrectS)
    Dpool = CalcValuesCompact(Xall, Dict, dictList)
    fd = {X:Xnor}
    #fd = {X:Xnor, ux:Dpool['ux'],uy:Dpool['uy'],uxx:Dpool['uxx'],uyy:Dpool['uyy']}

    s_est = sess.run([u], feed_dict=fd)
    s_est = np.squeeze(np.real(s_est))
    SEST = np.zeros_like(Dict['Omega'],dtype=np.float32)
    for i in range(Xall.shape[0]):
        y = Xall[i, 1] - 1
        x = Xall[i, 0] - 1
        SEST[y, x] = s_est[i]


    gx,gy = GetImageandGrads(SEST,Dict,key)
    [gxx,_] = GetImageandGrads(gx,Dict,key)
    [_, gyy] = GetImageandGrads(gy, Dict, key)

    return SEST,gx,gy,gxx,gyy


def CalcLoss(sigma,sx,sy,ux,uy,uxx,uyy,lam,lam_inf,BatchSize,K):

    div = (sx * ux + sy * uy + sigma * (uxx + uyy))

    topk = tf.nn.top_k(tf.reshape(tf.abs(div), (-1,)), K)
    loss_inf = lam_inf * tf.reduce_mean(topk.values)


    loss_l2 = lam * tf.reduce_mean(tf.square(div))


    return loss_inf,loss_l2


def Solve():

    with open('Data/Tempdata.json') as f:
        data = json.load(f)

    if data['useGPU']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    inputSize = data['inputSize']
    outputSize = data['outputSize']
    hiddenSize1 = data['hiddenSize1']
    hiddenSize2 = data['hiddenSize2']
    hiddenSize3 = data['hiddenSize3']
    hiddenSize4 = data['hiddenSize4']
    NLayers = data['NLayers']
    epochs = data['epochs']
    BatchSize = data['BatchSize']
    reg_constant = data['reg_constant']
    lr = data['lr']
    lam = data['lam']
    lrReduceEvery = data['lrReduceEvery']
    ReduceFactor = data['ReduceFactor']
    EarlyStop = data['EarlyStop']
    SaveModel=data['SaveModel']
    ReloadFile = data['ReloadFile']
    TestName = data['TestName']
    InputFile = data['InputFile']
    MaxNumOfPoints = data['MaxNumOfPoints']
    lam_inf = data['lam_inf']
    lamlp=data['lamlp']
    P=data['P']
    NumOfU = data['NumOfU']
    K = data['K']

    #Dict = ReadData('Data/' + InputFile)
    Dict = ReadData(os.path.join(data['input_path'], InputFile))

    if NLayers < 1 or NLayers > 4:
        print('Illegal NLayers')
        return
    if NumOfU < 1 or NumOfU > 4:
        print('Illegal NumOfU')
        return

    print(Dict.keys())
    Xall,XallRand = ExtractX(Dict)

    Nb = min(data['Nb'], Dict['Xb'].shape[0])

    Xpool = np.zeros((BatchSize, 2), dtype=np.int64)
    NumOfLoops = int(np.ceil(min(XallRand.shape[0],MaxNumOfPoints)/BatchSize))
    CorrectS = Dict['CorrectS']
    CorrectB = Dict['CorrectB']
    # CorrectS = float(0.5*(max(Xall[:,0])-min(Xall[:,0])))
    # CorrectB = float(min(Xall[:,0])+CorrectS)

    h = Dict['h']

    uxList =  Dict['uxList']
    uyList =  Dict['uyList']
    uxxList = Dict['uxxList']
    uyyList = Dict['uyyList']



    Dict.update({'ux1': uxList[0,0]/h})
    Dict.update({'uy1': uyList[0,0]/h})
    Dict.update({'uxx1': uxxList[0,0]/(h*h)})
    Dict.update({'uyy1': uyyList[0,0]/(h*h)})

    Dict.update({'ux2': uxList[0, 1] / h})
    Dict.update({'uy2': uyList[0, 1] / h})
    Dict.update({'uxx2': uxxList[0, 1] / (h * h)})
    Dict.update({'uyy2': uyyList[0, 1] / (h * h)})

    Dict.update({'ux3': uxList[0, 2] / h})
    Dict.update({'uy3': uyList[0, 2] / h})
    Dict.update({'uxx3': uxxList[0, 2] / (h * h)})
    Dict.update({'uyy3': uyyList[0, 2] / (h * h)})

    Dict.update({'ux4': uxList[0, 3] / h})
    Dict.update({'uy4': uyList[0, 3] / h})
    Dict.update({'uxx4': uxxList[0, 3] / (h * h)})
    Dict.update({'uyy4': uyyList[0, 3] / (h * h)})



    dictList = ['Omega', 'W', 'ux1','uy1','uxx1','uyy1','ux2','uy2','uxx2','uyy2','ux3','uy3','uxx3','uyy3','ux4','uy4','uxx4','uyy4']

    X = tf.placeholder(tf.float32, [None, inputSize])
    XB = tf.placeholder(tf.float32, [None, inputSize])
    w = tf.placeholder(tf.float32, [None, 1])
    ux1 = tf.placeholder(tf.float32, [None, 1])
    uxx1 = tf.placeholder(tf.float32, [None, 1])
    uy1 = tf.placeholder(tf.float32, [None, 1])
    uyy1 = tf.placeholder(tf.float32, [None, 1])
    ux2 = tf.placeholder(tf.float32, [None, 1])
    uxx2 = tf.placeholder(tf.float32, [None, 1])
    uy2 = tf.placeholder(tf.float32, [None, 1])
    uyy2 = tf.placeholder(tf.float32, [None, 1])
    ux3 = tf.placeholder(tf.float32, [None, 1])
    uxx3 = tf.placeholder(tf.float32, [None, 1])
    uy3 = tf.placeholder(tf.float32, [None, 1])
    uyy3 = tf.placeholder(tf.float32, [None, 1])
    ux4 = tf.placeholder(tf.float32, [None, 1])
    uxx4 = tf.placeholder(tf.float32, [None, 1])
    uy4 = tf.placeholder(tf.float32, [None, 1])
    uyy4 = tf.placeholder(tf.float32, [None, 1])



    if NLayers == 2:
        NN = Neural_Network2Layers(inputSize, hiddenSize1, hiddenSize2, outputSize)
    if NLayers == 3:
        NN = Neural_Network3Layers(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize,0)
    if NLayers == 4:
        NN = Neural_Network4Layers(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4,outputSize)

    sigma = ((NN.forward(X)))

    c1 = NN.forward(XB) - 1
    loss0 = tf.reduce_mean(tf.abs(c1))

    [sx,sy] = MyGrad(sigma,X)


    [loss_inf_1,loss_l2_1]  = CalcLoss(sigma, sx, sy, ux1, uy1, uxx1, uyy1, lam, lam_inf, BatchSize,K)
    [loss_inf_2, loss_l2_2] = CalcLoss(sigma, sx, sy, ux2, uy2, uxx2, uyy2, lam, lam_inf, BatchSize,K)
    [loss_inf_3, loss_l2_3] = CalcLoss(sigma, sx, sy, ux3, uy3, uxx3, uyy3, lam, lam_inf, BatchSize,K)
    [loss_inf_4, loss_l2_4] = CalcLoss(sigma, sx, sy, ux4, uy4, uxx4, uyy4, lam, lam_inf, BatchSize, K)

    if NumOfU == 1:
        loss_inf = loss_inf_1
        loss_l2 = loss_l2_1
    if NumOfU == 2:
        loss_inf = loss_inf_2
        loss_l2 = loss_l2_2
    if NumOfU == 3:
        loss_inf = loss_inf_3
        loss_l2 = loss_l2_3
    if NumOfU == 4:
        loss_inf = (loss_inf_1+loss_inf_2+loss_inf_3+loss_inf_4) / 4.0
        loss_l2 = (loss_l2_1 + loss_l2_2 + loss_l2_3+loss_l2_4) / 4.0


    losslp = lamlp*tf.reduce_mean((sx**2+sy**2)**(P/2))

    if NLayers == 3:
        reg_losses = tf.nn.l2_loss(NN.W1)+tf.nn.l2_loss(NN.W2)+tf.nn.l2_loss(NN.W3)+tf.nn.l2_loss(NN.W4)
    if NLayers == 2:
        reg_losses = tf.nn.l2_loss(NN.W1)+tf.nn.l2_loss(NN.W2)+tf.nn.l2_loss(NN.W3)
    if NLayers == 4:
        reg_losses = tf.nn.l2_loss(NN.W1) + tf.nn.l2_loss(NN.W2) + tf.nn.l2_loss(NN.W3) + tf.nn.l2_loss(NN.W4)+tf.nn.l2_loss(NN.W5)
    l2Loss = reg_losses*reg_constant


    loss = loss0+loss_l2+loss_inf+l2Loss+losslp



    optsAdam = tf.train.AdamOptimizer(learning_rate=lr)
    optimizersAdam = optsAdam.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        min_loss = 1e+20
        counter = 0

        optim = optimizersAdam

        Xball = np.zeros((Nb, 2), dtype=np.int64)
        Xball[:, 0] = np.transpose(Dict['Xb'][0:Nb])
        Xball[:, 1] = np.transpose(Dict['Yb'][0:Nb])
        Xballnor = SetX(Xball, inputSize, CorrectB, CorrectS)

        for i in range(epochs):


                total_loss = 0
                for j in range(NumOfLoops):
                    j1 = j * BatchSize
                    j2 = min(XallRand.shape[0], (j + 1) * BatchSize)

                    Xpool[0:(j2 - j1), 0] = np.transpose(XallRand[j1:j2,0])
                    Xpool[0:(j2 - j1), 1] = np.transpose(XallRand[j1:j2, 1])
                    Xnor = SetX(Xpool, inputSize, CorrectB, CorrectS)


                    Dpool = CalcValuesCompact(Xpool, Dict, dictList)


                    tf_dict = {X: Xnor, XB: Xballnor, w: Dpool['W'],
                                ux1: Dpool['ux1'], uy1: Dpool['uy1'],uxx1:Dpool['uxx1'],uyy1:Dpool['uyy1'],
                                ux2: Dpool['ux2'], uy2: Dpool['uy2'], uxx2: Dpool['uxx2'], uyy2: Dpool['uyy2'],
                                ux3: Dpool['ux3'], uy3: Dpool['uy3'], uxx3: Dpool['uxx3'], uyy3: Dpool['uyy3'],
                               ux4: Dpool['ux4'], uy4: Dpool['uy4'], uxx4: Dpool['uxx4'], uyy4: Dpool['uyy4']
                                }




                    _, l_1,l0,l1,l2 = sess.run([optim, loss,loss0,loss_l2,l2Loss], feed_dict=tf_dict)


                    total_loss += l_1




                print("Epoch {0}: {1:.4e} loss0: {2:.4e} loss1:{3:.4e} loss2:{4:.4e}".format(i, total_loss / min(XallRand.shape[0],MaxNumOfPoints) ,
                                                                                             l0,l1,l2  ))




                if i % lrReduceEvery == 0:
                    lr = lr * ReduceFactor
                if total_loss < min_loss:
                    counter = 0
                    min_loss = total_loss
                    print('**')
                else:
                    counter += 1
                if counter > EarlyStop:
                    print('Early stop {0:.4e}'.format(min_loss / min(XallRand.shape[0],MaxNumOfPoints) ))
                    break


        SEST, SEST_X, SEST_Y, _, _ = GetImages(sess,X,Xall, Dict, dictList, 'Sigma', sigma, inputSize, CorrectB, CorrectS)

        # Xnor = SetX(Xall, inputSize, CorrectB, CorrectS)
        # Dpool = CalcValuesCompact(Xall, Dict, dictList)
        # fd = {X: Xnor, ux: Dpool['ux'], uy: Dpool['uy'], uxx: Dpool['uxx'], uyy: Dpool['uyy']}
        # div_est = sess.run([div], feed_dict=fd)
        # div_est = np.squeeze(np.real(div_est))
        # DIVEST = np.zeros_like(Dict['Omega'], dtype=np.float32)
        # for i in range(Xall.shape[0]):
        #     y = Xall[i, 1] - 1
        #     x = Xall[i, 0] - 1
        #     DIVEST[y, x] = div_est[i]



        err = calcError(Dict['Sigma'], SEST, Dict['Omega'])

        plt.clf()

        plt.figure(1)
        plt.colorbar(plt.imshow(SEST,cmap='viridis'))
        plt.title('Reconstructed Sigma: error={:0.4f}'.format(err))




        plt.savefig(os.path.join(data['output_path'], TestName + '.png'), dpi=150)


        OutDict = {}
        OutDict.update({'Sest': SEST})
        OutDict.update({'data':data})
        OutDict.update({'CorrectB':CorrectB})
        OutDict.update({'CorrectS': CorrectS})
        #scipy.io.savemat('Data/Output/'+TestName+'.mat',OutDict)
        scipy.io.savemat(os.path.join(data['output_path'], TestName + '.mat'), OutDict)

        #plt.show()
        plt.close('all')
        return err



if __name__== "__main__":
    err = Solve()
    print('----')
    print(err)