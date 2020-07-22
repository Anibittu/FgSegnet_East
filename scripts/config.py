

'''
from FgSegNet_add_module import FgSegNet_v2_module
model= FgSegNet_v2_module
max_epoch=10
lr =1e-4
val_split= 0.2
batch_size =1
'''




'''
from FgSegNet_concat_module import FgSegNet_v2_module
model= FgSegNet_v2_module
max_epoch=10
lr =1e-4
val_split= 0.2
batch_size =1
'''


'''
from FgSegNet_east_add_model import FgSegNet_v2_module
model= FgSegNet_v2_module
max_epoch=10
lr =1e-4
val_split= 0.2
batch_size =1
weights_path= '/home/anish/FgSegNet_v2-master/FgSegNet_v2/SBI/models/EAST_IC15+13_model.h5'
'''


from FgSegnet_east_concat_model import FgSegNet_v2_module
model= FgSegNet_v2_module
max_epoch=10
lr =1e-4
val_split= 0.2
batch_size =1
weights_path= '/home/anish/FgSegNet_v2-master/FgSegNet_v2/SBI/models/EAST_IC15+13_model.h5'

