# Load the model
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import torch
from torch import nn

model_loaded = SamModel.from_pretrained('pre_trained')
processor = SamProcessor.from_pretrained("MedSAM", local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded.to(device)

# load data
loaded_dict = np.load('data/downsampled_metadata.npy', allow_pickle=True).item()
training = loaded_dict['training']
testing = loaded_dict['testing']

# transform image
def transform_image(individual_id):
    img_array = np.load(f'data/root_image_folder/{individual_id}.npz')
    origImage =  img_array[img_array.files[0]]
    origImage = origImage*255/max(origImage.flatten())
    image = Image.fromarray(np.uint8(origImage)).convert('RGB')
    return image

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


def obtain_predictions(list_of_individs):
    
    results = []
    # for each image:
    for ind in list_of_individs:
        print(ind['id'])
        image = transform_image(ind['id'])
        ind_results = {}
        ind_results['id'] = ind['id']
        ind_results['predicted_labels'] = []
        
        # for each label in the image:
        for label_array in ind['labels']:
            label = np.uint8(label_array)
            
            # prepare image + box prompt for the model
            prompt = get_bounding_box(label)
            inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_loaded(**inputs, multimask_output=False)
                
            
            # upsample output
            predicted_masks = nn.functional.interpolate(outputs.pred_masks.squeeze(1),
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False)
            
            # transfer to cpu:
            medsam_seg_prob = predicted_masks.cpu().numpy().squeeze()
            medsam_seg_prob = (medsam_seg_prob > 0.1).astype(np.float16)
            print(np.sum(medsam_seg_prob))

            # add to the individual dictionary
            ind_results['predicted_labels'].append(medsam_seg_prob)

        # append to results
        results.append(ind_results)

    return results

#training_results = obtain_predictions(training)
testing_results = obtain_predictions(testing)

#np.save('data/training_predictions.npy', training_results)
np.save('data/testing_predictions.npy', testing_results)