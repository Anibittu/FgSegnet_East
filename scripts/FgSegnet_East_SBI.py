"""
Created on monday 1 june 2020

@author: Anish Kumar
"""

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import numpy as np
import tensorflow as tf
import random as rn
import os, sys
from keras.preprocessing import image
# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

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
import config
from keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight
from keras.utils.data_utils import get_file

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
def getData(train_dir, dataset_dir, scene):
    
    void_label = -1.
    
    Y_list = glob.glob(os.path.join(train_dir,'*.png'))
    X_list = glob.glob(os.path.join(dataset_dir,'*.png' ))
    if len(Y_list)<=0 or len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
        
    # filter matched files        
    X_list_temp = []
    for i in range(len(Y_list)):
        Y_name = os.path.basename(Y_list[i])
        for j in range(len(X_list)):
            X_name = os.path.basename(X_list[j])            
            if (Y_name == X_name):
                X_list_temp.append(X_list[j])
                break
            
    X_list = X_list_temp
    
    if len(X_list)!=len(Y_list):
        raise ValueError('The number of X_list and Y_list must be equal.')
        
    # X must be corresponded to Y
    X_list = sorted(X_list)
    Y_list = sorted(Y_list)
    # process training images
    X = np.zeros((len(X_list), 420, 540, 3))
    Y = np.zeros((len(Y_list), 420, 540, 1))
    for i, fig in enumerate(X_list):
        x = image.load_img(fig , target_size=(420, 540))
        x = image.img_to_array(x).astype('float32')
        x=x/255
        X[i]=x
    for i, fig in enumerate(Y_list):
        x = kImage.load_img(fig , grayscale=True,target_size=(420, 540))
        x = kImage.img_to_array(x).astype('float32')
        x=x/255
        Y[i]=x    
    X = np.asarray(X)
    print(np.size(X))
    Y = np.asarray(Y)
    print(np.size(Y))
    # Shuffle the training data
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    
    # compute class weights
    cls_weight_list = []
    for i in range(Y.shape[0]):
        y = Y[i].reshape(-1)
        idx = np.where(y!=void_label)[0]
        if(len(idx)>0):
            y = y[idx]
        lb = np.unique(y) #  0., 1
        cls_weight = compute_class_weight('balanced', lb , y)
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0
        
        cls_weight_dict = {0:class_0, 1: class_1}
        cls_weight_list.append(cls_weight_dict)
        
    cls_weight_list = np.asarray(cls_weight_list)
    print(np.shape(Y))
    return[X, Y, cls_weight_list]


 
def train(data, scene, mdl_path, vgg_weights_path):
    
    ### hyper-params
    lr = config.lr
    val_split = config.val_split 
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    ###
    
    img_shape = np.shape(data[0][0])#(height, width, channel)
    model = config.model(lr, img_shape, scene, vgg_weights_path)
    model = model.initModel('SBI')

    # make sure that training input shape equals to model output
    input_shape = (img_shape[0], img_shape[1])
    output_shape = (model.output._keras_shape[1], model.output._keras_shape[2])
    assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', 
                                     save_weights_only=False, period=1)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto')
    model.fit(data[0],data[1], validation_split=val_split, epochs=max_epoch, batch_size=batch_size, 
              callbacks=[redu, early,mc], verbose=1, class_weight=data[2], shuffle = True)
    #model.summary() 
    model.save(mdl_path)
    del model, data, early, redu



# =============================================================================
# Main func
# =============================================================================

dataset = [ 'train' ]


main_dir = os.path.join('..', 'FgSegNet_v2')
#vgg_weights_path = config.weights_path
vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(vgg_weights_path):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')
main_mdl_dir = os.path.join(main_dir, 'SBI', 'models')
if not os.path.exists(main_mdl_dir):
    os.makedirs(main_mdl_dir)
   
for scene in dataset:
    print ('Training ->>> ' + scene)
    
    train_dir = os.path.join('..', 'training_sets', 'SBI2015_train', scene)
    dataset_dir = os.path.join('..', 'datasets', 'SBI2015_dataset', scene, 'input')

    mdl_path = os.path.join(main_mdl_dir, 'mdl_' + scene + '1'+'.h5')

    results = getData(train_dir, dataset_dir, scene)
    train(results, scene, mdl_path, vgg_weights_path)
    del results
