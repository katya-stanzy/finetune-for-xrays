import numpy as np
import transformers
from PIL import Image
import os


tr = np.load('data/meta_training_full_labels.npy', allow_pickle=True).item()

dataset_dict = {"image":[], "label":[]}

for i, label_path in enumerate(tr['label']):
    label_list = label_path.split('/')
    dir = '/'.join(label_list[:-1])
    label_dir = os.listdir(dir)
    if label_list[-1] in label_dir:
        label_array = np.load(label_path)
        label = Image.fromarray(np.uint8(label_array))

        img_array = np.load(tr['image'][i])
        origImage =  img_array[img_array.files[0]]
        origImage = origImage*255/max(origImage.flatten())  
        image = Image.fromarray(np.uint8(origImage)).convert('RGB')

        dataset_dict['image'].append(image)
        dataset_dict['label'].append(label)

from datasets import Dataset
dataset = Dataset.from_dict(dataset_dict)

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - 10 - np.random.randint(0, 10))
  x_max = min(W, x_max + 10 + np.random.randint(0, 10))
  y_min = max(0, y_min - 10 - np.random.randint(0, 10))
  y_max = min(H, y_max + 10 + np.random.randint(0, 10))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

from torch.utils.data import Dataset
class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
  
# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("MedSAM", local_files_only = True)

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=dataset, processor=processor)

# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
batch = next(iter(train_dataloader))

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("MedSAM", local_files_only = True)

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# from tqdm import tqdm
from statistics import mean
import torch
# from torch.nn.functional import threshold, normalize
from torch import nn

#Training loop
num_epochs = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
  
model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_dataloader:
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      predicted_masks = nn.functional.interpolate(predicted_masks,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False)

      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')


model.save_pretrained('pre_trained', from_pt=True)
# Save the model's state dictionary to a file
torch.save(model.state_dict(), "pre_trained/finetuned_checkpoints.pth")