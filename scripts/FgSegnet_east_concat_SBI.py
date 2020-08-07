
"""
Created on monday 1 june 2020

@author: Anish Kumar
"""

import numpy as np
import tensorflow as tf
import random as rn
import os, sys
from keras.preprocessing import image
# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
import datetime
from keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight
from FgSegnet_east_concat_model import FgSegNet_v2_module
from keras.utils.data_utils import get_file
import cv2
# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# =============================================================================
# Few frames, load into memory directly
# =============================================================================

def getData(Y_list,X_list):
    while True:
        for i in range(len(Y_list)):
            x = image.load_img(X_list[i] , target_size=(420, 540))
            x = image.img_to_array(x).astype('float32')
            x=x/255
            X = np.zeros((len(X_list), 420, 540, 3))
            Y = np.zeros((len(Y_list), 420, 540))
            x=np.asarray(x)
            x = np.expand_dims(x, axis=0)
            X=x
            y=kImage.load_img(Y_list[i] , grayscale=True,target_size=(420, 540))
            y=kImage.img_to_array(y).astype('float32')
            y=y/255
            y=np.asarray(y)
            y = np.expand_dims(y, axis=0)
            Y=y
            '''
            y=y.reshape(-1)
            lb = np.unique(y)
            cls_weight = compute_class_weight('balanced', lb , y)
            class_0 = cls_weight[0]
            class_1 = cls_weight[1]
            cls_weight_dict = [class_0,class_1]
            cls_weight_list = np.asarray(cls_weight_dict)
            '''
            yield[X,Y]
 
def train(data, val, scene, mdl_path, vgg_weights_path):
    
    ### hyper-params
    lr = 1e-4
    val_split = 0.2
    max_epoch = 100
    batch_size = 953
    ###
    
    #img_shape = np.shape(data[0][0])#(height, width,
    a,b=next(data)
    img_shape=420,540,3
    print(img_shape)
    model = FgSegNet_v2_module(lr, img_shape, scene, vgg_weights_path)
    model = model.initModel('SBI')
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    mc = keras.callbacks.ModelCheckpoint('/home/ug2017/min/17155014/weights/weightec{epoch:08d}.h5', 
                                     save_weights_only=False, period=5)
    logs_base_dir='/home/ug2017/min/17155014/FgSegnet_East/scripts/logs'
    os.makedirs(logs_base_dir, exist_ok=True)
    logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto')
    model.fit_generator(data, steps_per_epoch=batch_size,validation_data =val,validation_steps=2, epochs=max_epoch,callbacks=[early,mc,redu,tensorboard_callback],shuffle=True,class_weight=[0.582,0.418])
    #model.summary() 
    model.save(mdl_path)
    del model, data, early, redu



# =============================================================================
# Main func
# =============================================================================

dataset = [ 'train' ]


main_dir = os.path.join('..', 'FgSegNet_v2')

vgg_weights_path = '/home/ug2017/min/17155014/EAST_IC15+13_model.h5'
main_mdl_dir = '/home/ug2017/min/17155014/FgSegnet_East'
if not os.path.exists(main_mdl_dir):
    os.makedirs(main_mdl_dir)
   
for scene in dataset:
    print ('Training ->>> ' + scene)
    mdl_path = os.path.join(main_mdl_dir, 'mdl_' + scene + 'eastcnt'+'.h5')
    train_dir = '/home/ug2017/min/17155014/trainy'
    dataset_dir = '/home/ug2017/min/17155014/inputx'
    Y_list = glob.glob(os.path.join(train_dir,'*.png'))
    X_list = glob.glob(os.path.join(dataset_dir,'*.png' ))
    valy_dir = '/home/ug2017/min/17155014/valy'
    valx_dir = '/home/ug2017/min/17155014/valx'
    Y_val = glob.glob(os.path.join(valy_dir,'*.png'))
    X_val = glob.glob(os.path.join(valx_dir,'*.png' ))
    results = getData(Y_list,X_list)
    val =getData(Y_val,X_val)
    train(results, val,scene, mdl_path, vgg_weights_path)
    del results