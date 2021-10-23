import utils
import torch
from ssdconfig import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from torch.optim import SGD
from ssd import SSD, MultiBoxLoss
from trainer import train, eval
from tqdm import tqdm 

config = SSDConfig()
device = config.DEVICE

config.PATH_TO_ANNOTATIONS = "data/annotation.txt"
config.PATH_TO_IMAGES = "ShelfImages/"
config.PATH_TO_CHECKPOINT = "ckpt/checkpoint_ssd_1.pth.tar"

config.PRINT_FREQ = 35
config.VGG_BN_FLAG = True
config.TRAIN_BATCH_SIZE = 8
config.LEARNING_RATE = 0.001
config.USE_PRETRAINED_VGG = False
config.NUM_ITERATIONS_TRAIN = 4000 


# dataloader
df = get_dataframe(config.PATH_TO_ANNOTATIONS)
dataset_tr = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=True)
dataloader_tr = DataLoader(dataset_tr,
                           shuffle=True,
                           collate_fn=collate_fn,
                           batch_size=config.TRAIN_BATCH_SIZE,
                           num_workers=config.NUM_DATALOADER_WORKERS)

dataset_te = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False)
dataloader_te = DataLoader(dataset_te,
                           shuffle=True,
                           collate_fn=collate_fn,
                           batch_size=config.TRAIN_BATCH_SIZE,
                           num_workers=config.NUM_DATALOADER_WORKERS)


try:
    checkpoint = torch.load(config.PATH_TO_CHECKPOINT)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
except FileNotFoundError:
    print('PATH_TO_CHECKPOINT not specified in SSDConfig.\nMaking new model and optimizer.')
    start_epoch = 0
    model = SSD(config)
    model_parameters = utils.get_model_params(model)
    optimizer = SGD(params=[{'params': model_parameters['biases'], 'lr': 2 * config.LEARNING_RATE},
                        {'params': model_parameters['not_biases']}],
                        lr=config.LEARNING_RATE,
                        momentum=config.MOMENTUM,
                        weight_decay=config.WEIGHT_DECAY)

# move to device
model.to(device)
criterion = MultiBoxLoss(model.priors_cxcy, config).to(device)
# num epochs to train
epochs = config.NUM_ITERATIONS_TRAIN // len(dataloader_tr)
# epoch where LR is decayed
decay_at_epoch = [int(epochs*x) for x in config.DECAY_LR_AT]


# fooh!!!! :)
for epoch in tqdm(range(start_epoch, epochs)):
    if epoch in decay_at_epoch:
        utils.adjust_learning_rate(optimizer, config.DECAY_FRAC)
    train(dataloader_tr, model, criterion, optimizer, epoch)
    if (epoch%5 == 0):
      print('Model checkpoint.', end=' ' )
      utils.save_checkpoint(epoch, model, optimizer, config, config.PATH_TO_CHECKPOINT)
      print('Model Evaluation.', end=' :: ')
      print('mAP: ' + str(eval(model, dataloader_te, 0.6, 0.4)))

utils.save_checkpoint(epoch, model, optimizer, config, config.PATH_TO_CHECKPOINT)