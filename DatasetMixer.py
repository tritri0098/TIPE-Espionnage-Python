import os
from PIL import Image
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

dataset_name = "LandCoverNet"
images_folder = "ref_landcovernet_eu_v1_source_sentinel_2"

datasets_root_folder = "c:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/" # Chemin du dataset "brut" (allégé)
final_dataset_root_folder = "c:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/processed/"         # Chemin des dataset

# Couleurs des labels
masks_labels = {
    0: (0,0,0),       # No data,
    1: (0,0,255),     # Water
    2: (136,136,136), # Artifical Bare Ground
    3: (209,164,109), # Natural   Bare Ground
    4: (245,245,255), # Snow / Ice
    5: (214,76,43),   # Woody Vegetation
    6: (24,104,24),   # Non-woody Cultivated Vegatation
    7: (0,255,0)      # Non-woody Semi-Natural Vegatation
}

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
    labels = rio.open(f"C:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_labels/ref_landcovernet_eu_v1_labels_{id}/labels.tif").read(1)
    mask = Image.fromarray(labels.astype("uint8"))

    rgb_mask = Image.new('RGB', (mask.size[0], mask.size[1]))

    # Coloration du masque via les labels
    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            rgb_mask.putpixel((x,y), masks_labels[mask.getpixel((x,y))])

    # Mise à jour du dictionnaire
    bijection[id] = i
    i += 1

    chip = id[-2:]

    j = 0
    for name in images_id:
        if name.find(id) != -1:
            red   = rio.open(f"C:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B04.tif").read(1)
            green = rio.open(f"C:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B03.tif").read(1)
            blue  = rio.open(f"C:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/ref_landcovernet_eu_v1/ref_landcovernet_eu_v1_source_sentinel_2/{name}/B02.tif").read(1)

            rgb = np.dstack((red, green, blue))
            rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(int)

            rgb_path = final_dataset_root_folder + f"Tile {i}/images/"
            if not os.path.exists(rgb_path):
                os.makedirs(rgb_path)
            str_j = str(j).zfill(3)
            plt.imsave(rgb_path+f"image_part_{str_j}.jpg", rgb.astype('uint8'))

            mask_path = final_dataset_root_folder + f"Tile {i}/masks/"
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            rgb_mask.save(mask_path + f"image_part_{str_j}.png")  # Sauvegarde du masque

            j += 1


