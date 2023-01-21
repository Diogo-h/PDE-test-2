import numpy as np
import tensorflow as tf
import os
import copy
import time
import scipy.ndimage as sn
import ApproxUBasedOnS as Ap
from ApproxUBasedOnS import *



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
pwd = os.getcwd()

d = os.path.join(pwd,'..\\Data\\InputData')

with open('Data/test2.json') as f:
    data = json.load(f)
errVec = []

tmpl = 'L5H2'
n = 3
phase = 4

data['input_path'] = os.path.join(pwd,'../Data/InputData')
data['output_path'] = os.path.join(pwd,'../Data/OutputData')
data['hiddenSize1']=26
data['hiddenSize2']=26
data['hiddenSize3']=26
data['hiddenSize4']=10
data['lam']= 0.01
data['lam_inf'] = 0.01
data['lamlp'] = 1e-2
data['K'] = 40
data['lr']= 5e-3
data['NLayers']= 4
data['reg_constant'] = 1e-8

data['EarlyStop'] = 20#50
data['lrReduceEvery'] = 200
data['ReduceFactor'] = 0.8
data['MaxNumOfPoints'] = 80*1000
data['epochs']=500
data['BatchSize'] = 1000#1000
data['Nb'] = 1200
data['NumOfU'] = 1
data['useGPU'] = False


data['InputFile'] = 'Grid1k' + tmpl + 'n'+ str(n) + '_phase_' + str(phase) +'.mat'
data['TestName'] = 'Grid1k' + tmpl + 'n'+ str(n) + '_phase_' + str(phase) + '_output'

start_time = time.time()
print('-----New recipe-----')
print(data['TestName'] )
with open('Data/Tempdata.json', 'w') as fp:
    json.dump(data, fp)
err = Ap.Solve()
minu = (time.time() - start_time)/60.0
print('time={0:.2f} minutes'.format(minu))

