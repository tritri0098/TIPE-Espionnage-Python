import os
import sys
import time

import cv2
from PIL import Image
import numpy as np
#from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf # Attention on utilise tensorflow==2.4 car incompatiblité avec keras==2.11
from tensorflow.keras.utils import to_categorical
from keras.models import Model # Attention ! On utilise keras==2.4 car incompatibilité avec tensorflow==2.11
from keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda
from keras import backend as K # Keras = squelette du DCNN (haut-niveau, faibles performances) mais les calculs sont faits via la backend par du code bas-niveau (tensorflow par défaut)
from keras.models import load_model
import segmentation_models as sm # On utilise la version 1.0.1 et non pas la 0.2.1 ! ($pip install segmentation-models et non pas $pip install segmentation_models)

#np.set_printoptions(threshold=sys.maxsize)

print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

minmaxscaler = MinMaxScaler()

dataset_root_folder = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/"
dataset_name = "LandCoverNet"

# permet de vérifier qu'il y a autant d'images que de masques
'''counts_images = []
counts_masks = []

i = 1
k = 1'''

'''for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
    dir_name = path.split(os.path.sep)[-1]

    images = os.listdir(path)
    str_path = str(path).replace('\\', '/')
    if str_path.find('/images') != -1 or str_path.find('/masks') != -1:
        if dir_name == 'images':
            counts_images.append((len(images), path))
            i += 1
        else:
            counts_masks.append((len(images), path))
            k += 1

        for i, image_name in enumerate(images):
            if (image_name.endswith('.jpg')):
                a = True


print(counts_images)
print(counts_masks)

for j in range(0,len(counts_masks)):
    if counts_images[j][0] != counts_masks[j][0]:
        print(counts_images[j])'''

image_patch_size = 256

image_dataset = []
mask_dataset  = []

for image_type in ['images', 'masks']:
    if image_type == 'images':
        image_extension = 'jpg'
    else:
        image_extension = 'png'
    for tile_id in range(1,806):
        for image_id in range(0,40):
            image = cv2.imread(f'{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_{str(image_id).zfill(3)}.{image_extension}', 1)
            if image is not None:
                if image_type == 'masks': # Seulement pour les masques
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                size_x = (image.shape[1]//image_patch_size)*image_patch_size
                size_y = (image.shape[0]//image_patch_size)*image_patch_size

                image = Image.fromarray(image) # Attention ce n'est plus un tableau mais un Image
                image = image.crop((0, 0, size_x, size_y))

                image = np.array(image) # Attention ce n'est plus un image mais un ndarray

                if image_type == 'images':
                    image_dataset.append(image)
                else:
                    mask_dataset.append(image)

                '''patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)

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
                            mask_dataset.append(individual_patched_mask)'''

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

class_building = '#888888'
class_building = class_building.lstrip(('#'))
class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))

class_road = '#d1a46d' # route "naturelle"
class_road = class_road.lstrip(('#'))
class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))

class_snow = '#f5f5ff'
class_snow = class_snow.lstrip(('#'))
class_snow = np.array(tuple(int(class_snow[i:i+2], 16) for i in (0,2,4)))

class_vegetation_woody = '#d64c2b'
class_vegetation_woody = class_vegetation_woody.lstrip(('#'))
class_vegetation_woody = np.array(tuple(int(class_vegetation_woody[i:i+2], 16) for i in (0,2,4)))

class_vegetation_cultivated = '#186818'
class_vegetation_cultivated = class_vegetation_cultivated.lstrip(('#'))
class_vegetation_cultivated = np.array(tuple(int(class_vegetation_cultivated[i:i+2], 16) for i in (0,2,4)))

class_vegetation_natural = '#00ff00'
class_vegetation_natural = class_vegetation_natural.lstrip(('#'))
class_vegetation_natural = np.array(tuple(int(class_vegetation_natural[i:i+2], 16) for i in (0,2,4)))

class_water = '#0000ff'
class_water = class_water.lstrip(('#'))
class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))

class_unlabeled = '#000000'
class_unlabeled = class_unlabeled.lstrip(('#'))
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))

'''class_building = '#3C1098'
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
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))'''

def rgb_to_label(label):
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    label_segment[np.all(label == class_unlabeled, axis=-1)] = 0
    label_segment[np.all(label == class_water, axis=-1)] = 1
    label_segment[np.all(label == class_building, axis=-1)] = 2
    label_segment[np.all(label == class_road, axis=-1)] = 3
    label_segment[np.all(label == class_snow, axis=-1)] = 4
    label_segment[np.all(label == class_vegetation_woody, axis=-1)] = 5
    label_segment[np.all(label == class_vegetation_cultivated, axis=-1)] = 6
    label_segment[np.all(label == class_vegetation_natural, axis=-1)] = 7
    label_segment = label_segment[:,:,0]
    return label_segment

labels = []

for i in range(mask_dataset.shape[0]):
    label = rgb_to_label(mask_dataset[i])
    labels.append(label)

print(image_dataset.shape[0])

labels = np.array(labels)

print(labels)

labels = np.expand_dims(labels, axis=3)

total_classes = len(np.unique(labels))

labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)
master_training_dataset = image_dataset

x_train, x_test, y_train, y_test = train_test_split(master_training_dataset, labels_categorical_dataset, test_size=0.15, random_state=100)

image_height   = x_train.shape[1]
image_width    = x_train.shape[2]
image_channels = x_train.shape[3]
total_classes  = y_train.shape[3]

# Partie Deep Learning

def jaccard_coef(y_true, y_pred): # permet de rendre compte de l'accord entre le dataset et les prédictions (1 = parfait, 0 = complètement nul)
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

def multi_unet_model(n_classes=total_classes, image_height=image_height, image_width=image_width, image_channels=1):
    # On suit exactement le modèle ReLU
    inputs = Input((image_height, image_width, image_channels))

    source_input = inputs
    dropout_coef = 0.2  # On peut changer la valeur jusqu'à trouver celle qui minimise le coefficient de Jaccard

    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
    c1 = Dropout(dropout_coef)(c1)
    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(dropout_coef)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(dropout_coef)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(dropout_coef)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = Dropout(dropout_coef)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = Dropout(dropout_coef)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(dropout_coef)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = Dropout(dropout_coef)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3) # Attention : axis=3
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(dropout_coef)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

metrics = ["accuracy", jaccard_coef]

def get_deep_learning_model():
    return multi_unet_model(n_classes=total_classes, image_height=image_height, image_width=image_width, image_channels=image_channels)

model = get_deep_learning_model()


# Loss function

weights = [0.09642857142,0.09642857142,0.8,0.09642857142,0.09642857142,0.09642857142,0.09642857142,0.09642857142] # pour le nombre de poids et les poids se réferrer à la théorie sur le focal loss (pour l'approche uniforme choisir 1/N)
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss() # permet de plus pondérer les classes minoritaires dans les exemples que les classes faciles à identifier (=> moins de biais dûs à l'entropie croisée)
total_loss = dice_loss + (1 * focal_loss)

tf.keras.backend.clear_session()

#model.compile(optimizer="adam", loss=total_loss, metrics=metrics)

#print(model.summary()) # permet de voir les caractéristiques du modèle

# Entraînement du modèle
# Si possible utiliser l'accélération matérielle par GPU
# -> installer cuda toolkit

nb_epochs = 220 # nombre de passes (plus => meilleure fiabilité)

'''model_history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=False)

history_a = model_history

loss = history_a.history['loss']
val_loss = history_a.history['val_loss']
jaccard_coef = history_a.history['jaccard_coef']
val_jaccard_coef = history_a.history['val_jaccard_coef']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training Vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

epochs = range(1, len(jaccard_coef) + 1)

plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
plt.plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
plt.title("Training Vs Validation IoU")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Prédictions

y_pred = model.predict(x_test)

#print(len(y_pred))

# Comparaisons visuelles entre test et prédiction

y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

test_image_number = 4 #id de l'image à comparer

test_image = x_test[test_image_number]
ground_truth_image = y_test_argmax[test_image_number]

test_image_input = np.expand_dims(test_image, 0)

prediction = model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)

predicted_image = predicted_image[0,:,:]

plt.imshow(test_image)
plt.imshow(ground_truth_image)
plt.imshow(predicted_image)

# Sauevgarde du modèle

model_name = "landcovernet_model"
model.save(f"F:/Documents/Prepa/TIPE/espionnage/IA/models/{model_name}.h5")

#model.loss.name #permet de trouver le nom de la fonction perte pour bien paramétrer la fonction load_model

# Chargement du modèle

saved_model = load_model(f"F:/Documents/Prepa/TIPE/espionnage/IA/models/{model_name}.h5", custom_objects=({'dice_loss_plus_1focal_loss' : total_loss, 'jaccard_coef': jaccard_coef}))
#saved_model.get_config() # Permet de vérifier que le modèle chargé est cohérent (que c'est bien celui qui a été sauvegardé)

# Prédictions

image = Image.open("C:/Documents/Prepa/TIPE/espionnage/IA/datasets/testing/response.tiff")
orig = image

image = image.resize((256,256))
image = np.array(image)
image = np.expand_dims(image, 0)

predicted = saved_model.predict(image)
predicted_im = np.argmax(predicted, axis=3)
predicted_im = predicted_im[0,:,:]

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(orig)
plt.subplot(232)
plt.title("Predicted Image")
plt.imshow(predicted_im)'''

model_name = "landcovernet_model_2"

checkpoint_filepath = 'F:/Documents/Prepa/TIPE/espionnage/IA/models/checkpoints/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

saved_model = load_model(f"F:/Documents/Prepa/TIPE/espionnage/IA/models/{model_name}.h5", custom_objects=({'dice_loss_plus_1focal_loss' : total_loss, 'jaccard_coef': jaccard_coef}))
#saved_model.get_config() # Permet de vérifier que le modèle chargé est cohérent (que c'est bien celui qui a été sauvegardé)
tf.keras.backend.clear_session()
saved_model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
saved_model_history = saved_model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=False, callbacks=[model_checkpoint_callback])

saved_model.save(f"F:/Documents/Prepa/TIPE/espionnage/IA/models/{model_name}_3.h5")

loss = saved_model_history.history['loss']
val_loss = saved_model_history.history['val_loss']
jaccard_coef = saved_model_history.history['jaccard_coef']
val_jaccard_coef = saved_model_history.history['val_jaccard_coef']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training Vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

epochs = range(1, len(jaccard_coef) + 1)

plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
plt.plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
plt.title("Training Vs Validation IoU")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
