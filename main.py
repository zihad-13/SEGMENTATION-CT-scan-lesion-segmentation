import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time
from tqdm import tqdm 
from model import UNET
import torch.nn as nn
import torch.optim as optim

# %%
train_dir='data/frames'
train_maskdir='data/masks'
outdir='outputs/'

if not os.path.exists(outdir):
    os.makedirs(outdir)
# %%
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1280 originally
IMAGE_WIDTH = 128  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
num_workers=4
pin_memory=True
# %%
import pdb
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            
            # pdb.set_trace()
            # augmentations = self.transform(image=image, mask=mask)
            # image = augmentations["image"]
            # mask = augmentations["mask"]
            image = self.transform(image) 
            mask = self.transform(mask)
        return image, mask
    
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((IMAGE_HEIGHT,
		IMAGE_WIDTH)),
	transforms.ToTensor()])
    
    
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)   
datasets = Dataset(
    image_dir=train_dir,
    mask_dir=train_maskdir,
    transform=transforms,
)

loader = DataLoader(
    datasets,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
)

# %%

# Creating data indices for training and validation splits:
dataset_size = len(datasets)
indices = list(range(dataset_size))
split = 300
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE,
                                                sampler=valid_sampler)

# %%
# calculate steps per epoch for training and test set
trainSteps = int(dataset_size-split) // BATCH_SIZE
testSteps = int(split) // BATCH_SIZE
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#  store training history
storeLoss = {"train_loss": [], "validation_loss": []}
print("training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(train_loader):
		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = loss_fn(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in validation_loader:
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalTestLoss += loss_fn(pred, y)
            
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	storeLoss["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	storeLoss["validation_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("Total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
# %%
figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].set_title("training loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")

ax[1].set_title("validation loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("loss")

ax[0].plot(storeLoss["train_loss"],marker='x')
ax[1].plot(storeLoss["validation_loss"],marker='x')

plt.show()
# %%

def SegmentPlot(Image, gtMask, predMask,count,savefig=False):
    #pdb.set_trace()
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    ax[0].imshow(Image.reshape(Image.shape[1],Image.shape[2],Image.shape[0]))
    ax[1].imshow(gtMask.squeeze(0))
    ax[2].imshow(predMask.squeeze(0))
    ax[0].set_title("Image")
    ax[1].set_title("ground truth Mask")
    ax[2].set_title("Predicted Mask")
    figure.tight_layout()
    figure.show()
    if savefig: plt.savefig(outdir+str(count)+'.png')
model.eval()
count=1
for idx, (x, y) in enumerate(validation_loader):
    x = x.to(device=DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
        preds = preds.cpu().numpy()
        
        preds = preds.astype(np.uint8)
        for i in range(x.shape[0]):
            SegmentPlot(x[i].cpu().numpy(),y[i].cpu().numpy(),preds[i],count,savefig=True)
            count+=1
            





# %%

       
    


