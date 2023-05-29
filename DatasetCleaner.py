import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
import shutil

dataset_name = "LandCoverNet"
images_folder = "ref_landcovernet_eu_v1_source_sentinel_2"

datasets_root_folder = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/" # Chemin des dataset

p = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/ref_landcovernet_eu_v1_source_sentinel_2_39VXD_23_20180827"
cloud_trigger_val = 0.0 # pourcentage à partir duquel une image n'est plus admissible pour l'apprentissage automatique

i = 0
acc = 0

for path, subdirs, files in os.walk(os.path.join(datasets_root_folder, images_folder)):
    dir_name = path.split(os.path.sep)[-1]

    if i > 0: # path est invalide si i = 0
        str_path = str(path)
        subpath = (str_path + "/CLD.tif").replace('\\', '/')

        cloud_layer = Image.open(subpath)  # Image représentant la probabilité de densité nuageuse sur chaque pixel
        flag = False
        cloud_density = 100

        try:
            cloud_density = np.array(cloud_layer).mean()  # On calcule la moyenne des pixels de l'image (cela donne le pourcentage de couverture nuageuse)
        except:
            flag = True

        cloud_layer.close() # On ferme l'image pour éviter d'avoir des problèmes lors de la suppression de dossiers

        if cloud_density <= cloud_trigger_val and flag == False:
            acc += 1
            #print(f"n°{i} {cloud_density}: Acceptée")
        else:
            #print(f"n°{i} {cloud_density}: Refusée")
            corr_path = str_path.replace('\\', '/')
            shutil.rmtree(corr_path)  # Supprime le dossier qui ne vérifie pas les conditions nuageuses requises

        # On supprime les bandes inutiles (résolution >= 20m ou bandes de fréquences inutilisables)
        os.remove((str_path + "/B01.tif").replace('\\', '/')) # Supprime la bande "Coastal Aerosol"
        os.remove((str_path + "/B05.tif").replace('\\', '/'))  # Supprime la bande "Vegetation Red Edge (704.1nm)"
        os.remove((str_path + "/B06.tif").replace('\\', '/'))  # Supprime la bande "Vegetation Red Edge (740.1nm)"
        os.remove((str_path + "/B07.tif").replace('\\', '/'))  # Supprime la bande "Vegetation Red Edge (782.8nm)"
        os.remove((str_path + "/B8A.tif").replace('\\', '/'))  # Supprime la bande "Narrow NIR" (très proche infra-rouge)
        os.remove((str_path + "/B09.tif").replace('\\', '/'))  # Supprime la bande "Water Vapour"
        os.remove((str_path + "/B11.tif").replace('\\', '/'))  # Supprime la bande "SWIR (1613.7nm)"
        os.remove((str_path + "/B12.tif").replace('\\', '/'))  # Supprime la bande "SWIR (2202.4nm)"
        os.remove((str_path + "/SCL.tif").replace('\\', '/'))  # Supprime la bande "Scene Classification Layer"
        os.remove((str_path + "/CLD.tif").replace('\\', '/'))  # Supprime la bande "Cloud Mask" (désormais inutile car on a pris cloud_trigger_val = 0.0)

    i += 1

print(f"i = {i}")
print(f"acc = {acc}")
print(f"rej = {(i - acc)}")
print(f"Proportion acceptée : {((acc*100)/(i))} %")
