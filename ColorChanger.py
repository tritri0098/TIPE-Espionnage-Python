from PIL import Image

image_path = "C:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/Tile 790/masks/"
image_name = "image_part_000.png"

image = Image.open(image_path+image_name)

corrected_image = Image.new('RGB', (image.size[0], image.size[1]))

colors_to_correct = [(214,76,43), (24,104,24), (0,255,0), (209,164,109), (0,0,255)]
correct_color = (245,245,255) # White -> Snow / Ice

# Coloration du masque via les labels
for x in range(image.size[0]):
    for y in range(image.size[1]):
        current_pixel = image.getpixel((x,y))
        if current_pixel in colors_to_correct:
            corrected_image.putpixel((x,y), correct_color)
        else:
            corrected_image.putpixel((x, y), current_pixel)

corrected_image.save(image_path + image_name)  # Sauvegarde du masque

corrected_image.close()
image.close()
