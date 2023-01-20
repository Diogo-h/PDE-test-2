import numpy as np
import tensorflow as tf
import os
import copy
import time
import scipy.ndimage as sn


from ApproxSBasedOnMultU import *
import ApproxSBasedOnMultU as Ap
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


with open('Data/test2.json') as f:
    data = json.load(f)
errVec = []

pwd = os.getcwd()

tmpl = 'L5H2'
tmpl1 = 'Uest1k'
tmpl2 = 'Sest1k'


input_ = tmpl1+tmpl+'all.mat'
nm_=tmpl2+tmpl+'_1234'



data['input_path'] = os.path.join(pwd,'../Data/OutputData')
data['output_path'] = os.path.join(pwd,'../Data/OutputData')
data['hiddenSize1']=26
data['hiddenSize2']=26
data['hiddenSize3']=26
data['hiddenSize4']=10
data['lam']= 0.01
data['lam_inf'] = 1e-2
data['lamlp'] =1e-2
data['K']=40
data['P'] = 1.0
data['lr']= 5e-3
data['NLayers']= 4
data['reg_constant'] = 1e-8
data['InputFile'] = input_
data['EarlyStop'] = 50
data['lrReduceEvery'] = 200
data['ReduceFactor'] = 0.8
data['MaxNumOfPoints'] = 80*1000
data['epochs']=3000
data['BatchSize'] = 1000
data['Nb'] = 1200
data['useGPU'] = False
data['NumOfU'] = 1





data['NumOfU'] = 4
nm=nm_
print('-----New recipe-----')
print(nm)
data['TestName'] = nm
start_time = time.time()
with open('Data/Tempdata.json', 'w') as fp:
    json.dump(data, fp)
err= Ap.Solve()
errVec.append(err)
minu = (time.time() - start_time) / 60.0
print('time={0:.2f} minutes'.format(minu))



print('Results')
print(errVec)

