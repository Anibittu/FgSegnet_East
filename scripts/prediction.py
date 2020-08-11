import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from instance_normalization import InstanceNormalization
from my_upsampling_2d import MyUpSampling2D
from FgSegNet_add_module import loss, acc, loss2, acc2
import tensorflow as tf
import cv2
def load_image(path):
    x = image.load_img(path)
    x = image.img_to_array(x)
    x = cv2.resize(x, (int(540), int(420)))
    x = np.expand_dims(x, axis=0)
    return x

image_path = '/home/ug2017/min/17155014/inputx/0.png'
'''
for i in range (36):
    if((i%5==0)&(i>0)):
        if(i==5):
             model_path = '/home/ug2017/min/17155014/weights/weightconcat00000005.h5'
        else:
             model_path = '/home/ug2017/min/17155014/weights/weightconcat000000{}.h5'.format(i)
       
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'tf':tf,'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)
        plt.savefig('predFGadd{}.png'.format(i))
        plt.show()
'''
for i in range(5):
    if(i==0):
        model_path = '/home/ug2017/min/17155014/weights/weightconcat00000015.h5'
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask before thresholding')
        plt.axis('off')
        plt.savefig('try2.png')
        plt.show()
        # Thresholding (one can specify any threshold values)
        threshold = 0.8
        probs[probs<threshold] = 0.
        probs[probs>=threshold] = 1.
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask after thresholding')
        plt.axis('off')
        plt.savefig('TRY2.png')
        plt.show()
    if(i==1):
        model_path = '/home/ug2017/min/17155014/weights/weightconcat00000020.h5'
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask before thresholding')
        plt.axis('off')
        plt.savefig('try3.png')
        plt.show()
        # Thresholding (one can specify any threshold values)
        threshold = 0.8
        probs[probs<threshold] = 0.
        probs[probs>=threshold] = 1.
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask after thresholding')
        plt.axis('off')
        plt.savefig('TRY3.png')
        plt.show()
    if(i==2):
        model_path = '/home/ug2017/min/17155014/weights/weightconcat00000025.h5'
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask before thresholding')
        plt.axis('off')
        plt.savefig('try4.png')
        plt.show()
        # Thresholding (one can specify any threshold values)
        threshold = 0.8
        probs[probs<threshold] = 0.
        probs[probs>=threshold] = 1.
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask after thresholding')
        plt.axis('off')
        plt.savefig('TRY4.png')
        plt.show()
    if(i==3):
        model_path = '/home/ug2017/min/17155014/weights/weightconcat00000030.h5'
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask before thresholding')
        plt.axis('off')
        plt.savefig('try5.png')
        plt.show()
        # Thresholding (one can specify any threshold values)
        threshold = 0.8
        probs[probs<threshold] = 0.
        probs[probs>=threshold] = 1.
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask after thresholding')
        plt.axis('off')
        plt.savefig('TRY5.png')
        plt.show()
    if(i==4):
        model_path = '/home/ug2017/min/17155014/weights/weightconcat00000035.h5'
        x = load_image(image_path) # load a test frame
        model = load_model(model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2}) #load the trained model
        probs = model.predict(x, batch_size=1, verbose=1)
        print(probs.shape) # (1, 240,320,1)
        probs = probs.reshape([probs.shape[1], probs.shape[2]])
        print(probs.shape) # (240,320)
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask before thresholding')
        plt.axis('off')
        plt.savefig('try6.png')
        plt.show()
        # Thresholding (one can specify any threshold values)
        threshold = 0.8
        probs[probs<threshold] = 0.
        probs[probs>=threshold] = 1.
        plt.subplot(1, 1, 1)
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(probs)

        plt.title('Segmentation mask after thresholding')
        plt.axis('off')
        plt.savefig('TRY6.png')
        plt.show()
        
        
