import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

minmaxscaler = MinMaxScaler()

dataset_root_folder = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/"
dataset_name = "DubaiDataset"

for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
    dir_name = path.split(os.path.sep)[-1]

    if dir_name == 'images':
        images = os.listdir(path)

        '''for i, image_name in enumerate(images):
            if (image_name.endswith('.jpg')):
                a = True'''

image_patch_size = 256

image_dataset = []
mask_dataset  = []

for image_type in ['images', 'masks']:
    if image_type == 'images':
        image_extension = 'jpg'
    else:
        image_extension = 'png'
    for tile_id in range(1,8):
        for image_id in range(1,10):
            image = cv2.imread(f'{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}', 1)
            if image is not None:
                if image_type == 'masks': # Seulement pour les masques
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                size_x = (image.shape[1]//image_patch_size)*image_patch_size
                size_y = (image.shape[0]//image_patch_size)*image_patch_size

                image = Image.fromarray(image) # Attention ce n'est plus un tableau mais un Image
                image = image.crop((0, 0, size_x, size_y))

                image = np.array(image) # Attention ce n'est plus un image mais un ndarray
                patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)

                for i in range(patched_images.shape[0]):
                    for j in range(patched_images.shape[1]):


                        if image_type == 'images':
                            individual_patched_image = patched_images[i, j, :, :]

                            individual_patched_image = minmaxscaler.fit_transform(
                                individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(
                                individual_patched_image.shape)
                            individual_patched_image = individual_patched_image[0]

                            image_dataset.append(individual_patched_image)
                        else:
                            individual_patched_mask = patched_images[i, j, :, :]
                            individual_patched_mask = individual_patched_mask[0]
                            mask_dataset.append(individual_patched_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

class_building = '#3C1098'
class_building = class_building.lstrip(('#'))
class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))

class_land = '#8429F6'
class_land = class_land.lstrip(('#'))
class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))

class_road = '#6EC1E4'
class_road = class_road.lstrip(('#'))
class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))

class_vegetation = '#FEDD3A'
class_vegetation = class_vegetation.lstrip(('#'))
class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))

class_water = '#E2A929'
class_water = class_water.lstrip(('#'))
class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))

class_unlabeled = '#9B9B9B'
class_unlabeled = class_unlabeled.lstrip(('#'))
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))

def rgb_to_label(label):
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    label_segment[np.all(label == class_water, axis=-1)] = 0
    label_segment[np.all(label == class_land, axis=-1)] = 1
    label_segment[np.all(label == class_road, axis=-1)] = 2
    label_segment[np.all(label == class_building, axis=-1)] = 3
    label_segment[np.all(label == class_vegetation, axis=-1)] = 4
    label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
    label_segment = label_segment[:,:,0]
    return label_segment

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)

labels = np.expand_dims(labels, axis=3)

total_classes = len(np.unique(labels))

labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)
master_training_dataset = image_dataset

x_train, x_test, y_train, y_test = train_test_split(master_training_dataset, labels_categorical_dataset, test_size=0.15, random_state=100)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
