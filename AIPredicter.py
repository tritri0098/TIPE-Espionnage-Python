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
import segmentation_models as sm  # On utilise la version 1.0.1 et non pas la 0.2.1 ! ($pip install segmentation-models et non pas $pip install segmentation_models)

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

model_name = "landcovernet_model"

# Loss function

weights = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125] # pour le nombre de poids et les poids se réferrer à la théorie sur le focal loss (pour l'approche uniforme choisir 1/N)
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss() # permet de plus pondérer les classes minoritaires dans les exemples que les classes faciles à identifier (=> moins de biais dûs à l'entropie croisée)
total_loss = dice_loss + (1 * focal_loss)

def jaccard_coef(y_true, y_pred): # permet de rendre compte de l'accord entre le dataset et les prédictions (1 = parfait, 0 = complètement nul)
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

# Chargement du modèle

saved_model = load_model(f"F:/Documents/Prepa/TIPE/espionnage/IA/models/{model_name}_2.h5", custom_objects=({'dice_loss_plus_1focal_loss' : total_loss, 'jaccard_coef': jaccard_coef}))
#saved_model.get_config() # Permet de vérifier que le modèle chargé est cohérent (que c'est bien celui qui a été sauvegardé)

# Prédictions

image = Image.open("F:/Documents/Prepa/TIPE/espionnage/IA/datasets/testing/response.tiff")
ix = Image.open("F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/Tile 18/masks/image_part_009.png")
ix = np.array(ix)
ix = np.expand_dims(ix, 0)
orig = image

left = 600
top = 600

'''cropped = image.crop((left,top,left+256,top+256))
cropped = np.array(cropped)
cropped = np.expand_dims(cropped, 0)'''
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
plt.imshow(predicted_im)
