
"""
Created on Mon 1 june 2020

@author: Anish Kumar
"""

import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf
from keras.initializers import glorot_uniform
def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class FgSegNet_v2_module(object):
    
    def __init__(self, lr, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.method_name = 'FgSegNet_v2'
            
    def decoder(self,x,a,b):
        
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = concatenate([x, b])
        x =  concatenate([x, x1])
       
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = concatenate([x, a])
        x = concatenate([x, x2])
       
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
            
        x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        return x
    
    
    def M_FPM(self, x):
        
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)
        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)
        
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(y)
        
        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x
    
    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        print(len(self.img_shape))
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        net_input = Input(shape=(h, w, d), name='net_input')
        #vgg_output = self.VGG16(net_input)
        #Reading the model from JSON file
        with open('/home/anish/Downloads/model (copy).json', 'r') as json_file:
            json_savedModel= json_file.read()
        #load the model architecture 
        model_j = keras.models.model_from_json(json_savedModel,custom_objects= {'tf':tf,'RESIZE_FACTOR':2})
        #net_input=model_j(net_input)
        layer_output_1 = model_j.get_layer('conv1_pad').output
        layer_output_2 = model_j.get_layer('activation_1').output
        layer_output_3 = model_j.get_layer('activation_10').output
        model = Model(inputs=model_j.input, outputs=[layer_output_1,layer_output_2,layer_output_3], name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)
  
        layer_names=[layer.name for layer in model.layers]
        unfreeze_layers = []
       
        for i in layer_names:
                    layer_output = model.get_layer(i).output
                    unfreeze_layers.append(layer_output)
        
        for layer in model.layers:
            if(layer.name  in unfreeze_layers):
                layer.trainable = False   
       
        
        a,b,x = model(net_input)  
        print(a.shape,b.shape,x.shape)
        x = self.M_FPM(x)
        x = self.decoder(x,a,b)
        vision_model = Model(inputs=net_input, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)  
        c_loss = loss
        c_acc = acc
    
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model