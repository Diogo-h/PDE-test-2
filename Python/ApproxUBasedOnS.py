
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

    e = np.multiply(f-f_gt, Omega)
    err = np.mean(np.square(e))
    return err


def SetX(Xall,inputSize,CorrectB,CorrectS):

    Xnor = np.zeros([Xall.shape[0], inputSize])

    Xnor[:,0] = (Xall[:,0]-CorrectB)/CorrectS
    Xnor[:,1] = (Xall[:,1]-CorrectB)/CorrectS
    return Xnor


def GetImages(sess,X,Xall,Dict,dictList,key,u,inputSize,CorrectB,CorrectS):




    Xnor = SetX(Xall,inputSize,CorrectB,CorrectS)
    s_est = sess.run([u], feed_dict={X: Xnor })
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


    Dict = ReadData(os.path.join(data['input_path'],InputFile))
    N = Dict['U1'].shape[0]
    h = Dict['h']
    h_2 = h*h


    if NLayers < 1 or NLayers > 4:
        print('Illegal NLayers')
        return
    print(Dict.keys())
    Xall,XallRand = ExtractX(Dict)

    Nb = min(data['Nb'], Dict['Xb'].shape[0])


    Xpool = np.zeros((BatchSize, 2), dtype=np.int64)
    NumOfLoops = int(np.ceil(min(XallRand.shape[0],MaxNumOfPoints)/BatchSize))
    CorrectS = Dict['CorrectS']
    CorrectB = Dict['CorrectB']


    dictList = ['U1', 'Omega', 'W', 'Sigma', 'sx', 'sy']


    X = tf.compat.v1.placeholder(tf.float32, [None, inputSize])
    XB = tf.compat.v1.placeholder(tf.float32, [None, inputSize])
    w = tf.compat.v1.placeholder(tf.float32, [None, 1])
    u0 = tf.compat.v1.placeholder(tf.float32, [None, 1])
    sigma = tf.compat.v1.placeholder(tf.float32, [None, 1])
    sx = tf.compat.v1.placeholder(tf.float32, [None, 1])
    sy = tf.compat.v1.placeholder(tf.float32, [None, 1])




    if NLayers == 1:
        NN = Neural_Network1Layer(inputSize, hiddenSize1, outputSize)
    if NLayers == 2:
        NN = Neural_Network2Layers(inputSize, hiddenSize1, hiddenSize2, outputSize)
    if NLayers == 3:
        NN = Neural_Network3Layers(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize,0)
    if NLayers == 4:
        NN = Neural_Network4Layers(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize)

    c1 = NN.forward(XB) - u0
    loss0 = tf.reduce_mean(tf.abs(c1))
    #loss0 = tf.reduce_mean(tf.square(c1))




    u = NN.forward(X)



    [ux,uy] = MyGrad(u,X)
    [uxx,_] = MyGrad(ux,X)
    [_,uyy]=MyGrad(uy,X)

    div = (sx*ux/h+sy*uy/h+sigma*(uxx+uyy))



    K=data['K']
    topk = tf.nn.top_k(tf.reshape(tf.abs(div), (-1,)), K)
    loss_inf = lam_inf*tf.reduce_mean(topk.values)

    loss_l2 =  lam * tf.reduce_mean(tf.square(div))


    if NLayers == 3:
        reg_losses = tf.nn.l2_loss(NN.W1)+tf.nn.l2_loss(NN.W2)+tf.nn.l2_loss(NN.W3)+tf.nn.l2_loss(NN.W4)
    if NLayers == 2:
        reg_losses = tf.nn.l2_loss(NN.W1)+tf.nn.l2_loss(NN.W2)+tf.nn.l2_loss(NN.W3)
    if NLayers == 4:
        reg_losses = tf.nn.l2_loss(NN.W1) + tf.nn.l2_loss(NN.W2) + tf.nn.l2_loss(NN.W3) + tf.nn.l2_loss(NN.W4)+tf.nn.l2_loss(NN.W5)


    loss_regl2 = reg_losses*reg_constant


    loss = loss0+loss_inf+loss_l2+loss_regl2

    optsAdam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    optimizersAdam = optsAdam.minimize(loss)


    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        min_loss = 1e+20
        counter = 0

        Xball = np.zeros((Nb, 2), dtype=np.int64)
        Xball[:, 0] = np.transpose(Dict['Xb'][0:Nb])
        Xball[:, 1] = np.transpose(Dict['Yb'][0:Nb])
        Xballnor = SetX(Xball, inputSize, CorrectB, CorrectS)



        LOSS0_V = list()
        LOSS2_V = list()
        LOSSinf_V = list()
        LOSSreg_V = list()

        for i in range(epochs):

                optim = optimizersAdam

                total_loss = 0
                for j in range(NumOfLoops):
                    j1 = j * BatchSize
                    j2 = min(XallRand.shape[0], (j + 1) * BatchSize)

                    Xpool[0:(j2 - j1), 0] = np.transpose(XallRand[j1:j2,0])
                    Xpool[0:(j2 - j1), 1] = np.transpose(XallRand[j1:j2, 1])

                    Dpool = CalcValuesCompact(Xpool, Dict, dictList)

                    Xnor = SetX(Xpool,inputSize,CorrectB, CorrectS)

                    tf_dict = {X: Xnor, XB: Xballnor,u0: Dict['Ub'][0:Nb], w: Dpool['W'], sigma: Dpool['Sigma'],
                                                sx: Dpool['sx'], sy: Dpool['sy']}

                    _, l,l0,l2,linf,lreg = sess.run([optim, loss,loss0,loss_l2,loss_inf,loss_regl2], feed_dict=tf_dict)



                    total_loss += l

                LOSS0_V.append(l0)
                LOSS2_V.append(l2)
                LOSSinf_V.append(linf)
                LOSSreg_V.append(lreg/BatchSize)
                print("Epoch {0}: {1:.4e} loss0: {2:.4e} loss_l2:{3:.4e} loss_inf:{4:.4e} loss_reg:{5:.4e}".format(
                    i, total_loss / NumOfLoops,l0,l2,linf,lreg/BatchSize))




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


        UEST, UEST_X, UEST_Y, UEST_XX, UEST_YY = GetImages(sess,X,Xall, Dict, dictList, 'U1', u,inputSize, CorrectB, CorrectS)
        UEST_ResDerX, _, _, _, _ = GetImages(sess, X,Xall, Dict, dictList, '', (ux) * h,inputSize, CorrectB, CorrectS)
        UEST_ResDerXX, _, _, _, _ = GetImages(sess, X, Xall, Dict, dictList, '', (uxx) * h_2, inputSize, CorrectB, CorrectS)
        UEST_ResDerY, _, _, _, _ = GetImages(sess, X, Xall, Dict, dictList, '', (uy) * h, inputSize, CorrectB, CorrectS)
        UEST_ResDerYY, _, _, _, _ = GetImages(sess, X, Xall, Dict, dictList, '', (uyy) * h_2, inputSize, CorrectB,CorrectS)

        plt.clf()



        plt.figure(1)
        err = calcError(Dict['U1'], UEST, Dict['Omega'])

        plt.subplot(231)
        plt.imshow(UEST - Dict['U1'],cmap='viridis')
        plt.title("U-U_gt: error={:0.4f}".format(err))

        plt.subplot(232)
        # plt.colorbar(plt.imshow(UEST))
        plt.imshow(UEST,cmap='viridis')
        plt.title('U')

        plt.subplot(233)
        # plt.colorbar(plt.imshow(Dict['U1']))
        plt.imshow(Dict['U1'],cmap='viridis')
        plt.title('U_gt')

        # plt.figure(1)
        errux = calcError(Dict['ux'], UEST_ResDerX, Dict['Omega'])
        #errux = np.sum(np.square((UEST_ResDerX - Dict['ux'])*(1-Dict['W']))) / Nall
        plt.subplot(234)
        # plt.colorbar(plt.imshow(UEST_ResDerX-Dict['ux']))
        plt.imshow((UEST_ResDerX - Dict['ux'])*(1-Dict['W']),cmap='viridis')
        plt.title("Ux-Ux_gt: error= {:0.4e}".format(errux))

        plt.subplot(235)
        # plt.colorbar(plt.imshow(UEST_ResDerX))
        plt.imshow(UEST_ResDerX,cmap='viridis')
        plt.title("Ux")

        plt.subplot(236)
        # plt.colorbar(plt.imshow(Dict['ux']))
        plt.imshow(Dict['ux'],cmap='viridis')
        plt.title("Ux_gt")

        plt.savefig(os.path.join(data['output_path'],TestName+'.png'),dpi=150)

        OutDict = {}
        OutDict.update({'UEST': UEST})
        OutDict.update({'UESTx':UEST_ResDerX})
        OutDict.update({'UESTy': UEST_ResDerY})
        OutDict.update({'UESTxx': UEST_ResDerXX})
        OutDict.update({'UESTyy': UEST_ResDerYY})
        OutDict.update({'err': err})
        OutDict.update({'errux': errux})
        OutDict.update({'data':data})


        scipy.io.savemat(os.path.join(data['output_path'], TestName+'.mat'), OutDict)

        # plt.figure(2)
        # plt.semilogy(LOSS0_V,'b-.',LOSS2_V,'r--',LOSSinf_V, 'g-', LOSSreg_V, 'm-', linewidth=4)
        # plt.legend(['$L_1(\partial\Omega)$','$L_2$','$L_\infty$', '$L_{reg}$'], prop={'size': 14})
        # # plt.semilogy(LOSS0_V, 'b-.', LOSS2_V, 'r--', LOSSinf_V, 'g-',  linewidth=4)
        # # plt.legend(['$L_1(\partial\Omega)$', '$L_2$', '$L_\infty$'], prop={'size': 14})
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.savefig(os.path.join(data['output_path'], TestName + '_graph.png'), dpi=150)
        # plt.show()
        #
        # outPlot = {}
        # outPlot.update({'loss0':LOSS0_V})
        # outPlot.update({'loss2': LOSS2_V})
        # outPlot.update({'lossinf': LOSSinf_V})
        # outPlot.update({'lossreg': LOSSreg_V})
        # scipy.io.savemat(os.path.join(data['output_path'], TestName + '_plot.mat'), outPlot)


        return err



if __name__== "__main__":
    err = Solve()
    print('----')
    print(err)