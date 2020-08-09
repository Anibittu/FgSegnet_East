import numpy as np
from keras.preprocessing import image
from keras.models import load_model
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
#image_path ="/home/anish/Downloads/resize-15897880709460665360.png"

#model_path = 'mdl_highway_fgsegnet_v2.h5'
model_path = '/home/ug2017/min/17155014/weights/weightfad00000030.h5'

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

#plt.title('Segmentation mask before thresholding')
#plt.axis('off')
plt.savefig('/home/ug2017/min/17155014/FgSegnet_East/pred1.png')
plt.show()