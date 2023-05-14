from sentinelhub import SHConfig
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import wptools
import math
from geopy.geocoders import Nominatim
import sys
import mysql.connector
from PIL import Image as im
from numpy import asarray
np.set_printoptions(threshold=sys.maxsize)


config = SHConfig()  # Configuration API client de Sentinel Hub

config.instance_id = 'SatelliteExtractor'
config.sh_client_id = '51f0c297-a678-468a-a626-3d6c591949de'
config.sh_client_secret = 'F<mv8zJWwx8!(GiecSDn%[,eUy6et8iR?*4>zvks'

if not config.sh_client_id or not config.sh_client_secret:
    print("Les codes Sentinel Hub ne sont pas valides !\n")

time_int = 1461  # Intervalle (en jours pour chercher les photos avec le moins de nuages)

target_country = 'France'.lower()

embassies_list = [] # Liste des informations sur chaque ambassade. Format : [pays cible, pays hôte de l'ambassade, ville, adresse, lat, lng]


# Partie SQL

db = mysql.connector.connect(host="localhost", user="root", password="", database="embassies") # Création de la connexion avec la base de données
cursor = db.cursor()

cursor.execute('''SELECT host_country, city, latitude, longitude FROM embassies WHERE country="'''+target_country+'''"''')
rows = cursor.fetchall()
for row in rows:
   embassies_list.append(row)

db.commit() # Sauvegarde des changements sur la base de données

db.close() # Fin de la session de connexion à la base

# Méthodes de conversions de coordonées géographiques en WGS84 (EPSG:4326) géoide

# degrees to radians<
def deg_to_rad(deg):
    return math.pi * deg / 180.0


# radians to degrees
def rad_to_deg(rad):
    return 180.0 * rad / math.pi


# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]


# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))


# La surface de la Terre est considérée comme localement sphérique de rayon donné par la norme WGS84

def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):  # fonction calculant la bounding box en fonction des coordonées et de la demi largeur de bounding box en km
    lat = deg_to_rad(latitudeInDegrees)
    lon = deg_to_rad(longitudeInDegrees)
    halfSide = 1000 * halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)

    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    lonMin = lon - halfSide / pradius
    lonMax = lon + halfSide / pradius

    return (rad_to_deg(lonMin), rad_to_deg(latMin), rad_to_deg(lonMax), rad_to_deg(latMax))

count = 1 # On compte le nombre de villes traîtées
total = len(embassies_list)

for embassy in embassies_list:

    city = embassy[1] # Nom de la ville en minuscule

    # détermination de la longueur typique de la ville au format carré
    typical_length = 10.0  # valeur par défaut
    min_typical_length = 3.0  # valeur en dessous de laquelle la ville est trop petite au vu de son importance (erreur des unités sur wikipédia)

    try:
        pageData = wptools.page(city).get_parse()
        # Affinage de la valeur si possible
        infobox_keys = pageData.infobox.keys()
        if "area_total_km2" in infobox_keys:
            typical_length = math.sqrt(float(pageData.infobox["area_total_km2"].replace(',', '.')))
        elif "area km2" in infobox_keys:
            typical_length = math.sqrt(float(pageData.infobox["area km2"].replace(',', '.')))
        elif "area_urban_km2" in infobox_keys:  # Plus grossier la zone périurbaine étant généralement beaucoup plus grande
            typical_length = math.sqrt(float(pageData.infobox["area_urban_km2"].replace(',', '.')))
        elif "area_blank2_km2" in infobox_keys:  # En dernier recours ...
            typical_length = math.sqrt(float(pageData.infobox["area_blank2_km2"].replace(',', '.')))

        if typical_length < min_typical_length:  # S'il y a eu erreur d'unité de la part de wikipédia
            typical_length *= 10 * math.sqrt(
                10)  # facteur correctif de 10 * sqrt(10) pour retomber sur la bonne unité (si wikipédia s'est trompé, il a mis 1,000 km² au lieu de 1000 km²)
    except:
        print("Pas d'affinage possible pour cette ville")

    # Conversion des coordonnées comme prévu

    city_coords_wgs84 = (boundingBox(embassy[2], embassy[3], typical_length))

    # Adaptation de la bounding box à la résolution imposée

    resolution = (typical_length * 1000) / 2500  # première approximation

    city_bbox = BBox(bbox=city_coords_wgs84, crs=CRS.WGS84)
    city_size = bbox_to_dimensions(city_bbox, resolution=resolution)

    while max(
            city_size) > 2500:  # Tant qu'il y a un dépassement de taille ... (on veut max 2500px en largeur ou hauteur)
        resolution *= ((max(city_size)) / 2500)  # Affinage
        city_size = bbox_to_dimensions(city_bbox, resolution=resolution)

    # Requêtes d'acquisition d'images en couleurs "vraies"

    evalscript_true_color = """
                //VERSION=3

                function setup() {
                    return {
                        input: [{
                            bands: ["B02", "B03", "B04"]
                        }],
                        output: {
                            bands: 3
                        }
                    };
                }

                gain = 3.5

                function evaluatePixel(sample) {
                    return [sample.B04 * gain, sample.B03 * gain, sample.B02 * gain];
                }
            """
    # le gain sert à éclaircir les images qui ont tendances à êtres beaucoup trop sombres

    evalscript_dem = """
                //VERSION=3

                function setup() {
                  return {
                    input: [{
                        bands:["DEM"],
                        units:"DN"
                    }],
                    output:{
                        id: "default",
                        bands: 1,
                        sampleType: SampleType.FLOAT32
                    }
                  }
                }

                gain = 0.005

                function evaluatePixel(sample) {
                  return [sample.DEM*gain]
                }
            """
    # le gain sert à éclaircir les images qui ont tendances à êtres beaucoup trop claires

    # Gestion de l'intervalle de temps d'acuisition des images
    today = datetime.date.today()
    prec_date = today - datetime.timedelta(days=time_int)

    # On récupère l'image en couleur
    request_true_color = SentinelHubRequest(
        data_folder="C:/Users/migno/Desktop/satellite/" + target_country + "/" + embassy[0] + "/true_color/",
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,  # alternative : SENTINEL2_L1C (couleurs vraies, moins de contraste)
                time_interval=(prec_date, today),
                mosaicking_order=MosaickingOrder.LEAST_CC
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=city_bbox,
        size=city_size,
        config=config,
    )

    true_color_imgs = request_true_color.get_data(save_data=True)
    color_data = true_color_imgs[0]

    # Maintenant on récupère les données altimétriques
    request_dem = SentinelHubRequest(
        data_folder="C:/Users/migno/Desktop/satellite/" + target_country + "/" + embassy[0] + "/dem/",
        evalscript=evalscript_dem,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM_COPERNICUS_30,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=city_bbox,
        size=city_size,
        config=config,
    )

    dem_imgs = request_dem.get_data(save_data=True)
    dem_data = dem_imgs[0]

    # Création du modèle 3D
    dem_heights = []
    dem_materials = []

    len_y, len_x = dem_data.shape
    for i in dem_data:
        line = []
        for j in i:
            line.append(j)
        dem_heights.append(line)

    # Forêts, végétations, champ plantés
    avg_green_field = (25, 46, 18)  # D'après échantillon photoshop des images satellite
    avg_green_forest = (32, 29, 17)  # Forêt moyenne marron-verte
    # Champs à nu, terre
    avg_brown_field1 = (102, 72, 40)  # Champ nu marron foncé
    avg_brown_field2 = (197, 156, 108)  # Champ nu marron clair
    avg_brown_field3 = (190, 149, 97)  # Champ nu marron moyen
    # Eaux, fleuves, rivières
    avg_green_canal = (22, 36, 22)  # Canal

    count += 1

    print(str(round(count*100/total, 2))+'%')
