import os
from PIL import Image
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

dataset_name = "LandCoverNet"
images_folder = "ref_landcovernet_eu_v1_source_sentinel_2"

datasets_root_folder = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/" # Chemin du dataset "brut" (allégé)
final_dataset_root_folder = "F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/processed/"         # Chemin des dataset

images_id = []
sentinel_tiles_id = []

# Bornes de recherche
beg = len("ref_landcovernet_eu_v1_source_sentinel_2_")
end = beg+len("30VUJ_08") # Exemple de format dan le nom de dossier

first = True # Marqueur de début de boucle

for path, subdirs, files in os.walk(os.path.join(datasets_root_folder, images_folder)):
    if first: # Le premier passage de boucle n'apporte rien
        first = False
    else:
        name = str(os.path.basename(os.path.normpath(str(path))))  # Récupère le nom du dernier dossier du chemin
        images_id.append(name)
        tile_id = name[beg:end]
        if not tile_id in sentinel_tiles_id:
            sentinel_tiles_id.append(name[beg:end])

i = 0
bijection = {}

for id in sentinel_tiles_id:
    labels = rio.open(f"F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_labels/ref_landcovernet_eu_v1_labels_{id}/labels.tif").read(1)
    mask = Image.fromarray(labels.astype("uint8"))

    # Mise à jour du dictionnaire
    bijection[id] = i
    i += 1

    chip = id[-2:]

    mask_path = final_dataset_root_folder + f"Tile {i}/Chip {chip}/masks/"
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    mask.save(mask_path+f"{i}.png")  # Sauvegarde du masque

    j = 0
    for name in images_id:
        if name.find(id) != -1:
            red   = rio.open(f"F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B04.tif").read(1)
            green = rio.open(f"F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B03.tif").read(1)
            blue  = rio.open(f"F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B02.tif").read(1)

            rgb = np.dstack((red, green, blue))
            rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(int)

            rgb_path = final_dataset_root_folder + f"Tile {i}/Chip {chip}/images/"
            if not os.path.exists(rgb_path):
                os.makedirs(rgb_path)
            plt.imsave(rgb_path+f"{id}_{j}.jpg", rgb.astype('uint8'))

            j += 1
