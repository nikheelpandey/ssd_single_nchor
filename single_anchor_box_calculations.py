        
        
        
        
        



INPUT_IMAGE_SIZE = 300
NUM_CLASSES = 2



# ------
# device
# ------
# DEVICE = torch.device('cuda:0') if torch.cudapython.is_available() else torch.device('cpu')

# -------------- --- --- --------
# Configurations for VGG Backbone
# -------------- --- --- --------
VGG_BN_FLAG = False # whether vgg is used with or without batch norm.
USE_PRETRAINED_VGG = False
VGG_BASE_IN_CHANNELS = 3
VGG_BASE_CONFIG = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
            'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
            'C', 512, 512, 512, 'M', 512, 512, 512]}

# reshaped parameters dimensions for VGG's conv 6 & 7
VGG_BASE_CONV67_VIEWS = [[4096, 512, 7, 7],
                                [4096],
                                [4096, 4096, 1, 1],
                                [4096]]
# subsampling ratio for VGG's conv 6 & 7
VGG_BASE_CONV67_SUBSAMPLE_FACTOR = [[4, None, 3, 3],
                                            [4],
                                            [4, 4, None, None],
                                            [4]]

# These index are used to stop forward pass loop at conv4_3
# and get it's features.
VGGBN_BASE_CONV43_INDEX = 33  # index of 'conv4_3' layer in VGG16_BN
VGG_BASE_CONV43_INDEX = 23  # index of 'conv4_3' layer in VGG16

# -------------- --- ---------- ------------
# Configurations for Auxiliary convolutions
# -------------- --- ---------- ------------
AUX_BASE_IN_CHANNELS = 1024
AUX_BASE_CONFIG = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]}
# ------- ----
# Feature Maps
# ------- ----
FM_NAMES = ['conv4_3', 'conv7', 'conv8_2',
                'conv9_2', 'conv10_2', 'conv11_2']
FM_NUM_CHANNELS = [ 512, 1024, 512, 256, 256, 256 ]
FM_DIMS = [38, 19, 10, 5, 3, 1]
FM_SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

FM_ASPECT_RATIO =       [[1.],
                        [1.],
                        [1.],
                        [1.],
                        [1.],
                        [1.]]

# # ith additional scale is geometric mean of scales of ith and (i+1)th FM.
# # Aspect Ratio for the priors corresponding to these scales is 1.
FM_ADDITIONAL_SCALES = [0.1414, 0.2738, 0.4541, 0.6314, 0.8077, 1.0]

# the +1 is for the additional scale        
# _c = 1 if len(FM_ADDITIONAL_SCALES)==len(FM_NAMES) else 0       
NUM_PRIOR_PER_FM_CELL = [len(x) for x in FM_ASPECT_RATIO]
print(NUM_PRIOR_PER_FM_CELL)