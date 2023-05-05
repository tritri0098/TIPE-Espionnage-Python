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

config = SHConfig() # Configuration API client de Sentinel Hub

config.instance_id = 'SatelliteExctractor'
config.sh_client_id = '51f0c297-a678-468a-a626-3d6c591949de'
config.sh_client_secret = 'F<mv8zJWwx8!(GiecSDn%[,eUy6et8iR?*4>zvks'

if not config.sh_client_id or not config.sh_client_secret:
    print("Les codes Sentinel Hub ne sont pas valides !\n")


city = 'Karkov' # Ville à scanner
time_int = 700  # en jours

# Méthodes de conversions de coordonées géographiques en WGS84 (EPSG:4326) géoide

# degrees to radians
def deg_to_rad(deg):
    return math.pi*deg/180.0
# radians to degrees
def rad_to_deg(rad):
    return 180.0*rad/math.pi

# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# La surface de la Terre est considérée comme localement sphérique de rayon donné par la norme WGS84

def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm): # fonction calculant la bounding box en fonction des coordonées et de la demi largeur de bounding box en km
    lat = deg_to_rad(latitudeInDegrees)
    lon = deg_to_rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    return (rad_to_deg(latMin), rad_to_deg(lonMin), rad_to_deg(latMax), rad_to_deg(lonMax))

pageData = wptools.page(city).get_parse()

# détermination de la longueur typique de la ville au format carré
typical_length = 10.0 #valeur par défaut
# Affinage de la valeur si possible
infobox_keys = pageData.infobox.keys()
if "area_total_km2" in infobox_keys:
    typical_length = math.sqrt(float(pageData.infobox["area_total_km2"]))
elif "area km2" in infobox_keys:
    typical_length = math.sqrt(float(pageData.infobox["area km2"]))
elif "area_urban_km2" in infobox_keys: # Plus grossier la zone périurbaine étant généralement beaucoup plus grande
    typical_length = math.sqrt(float(pageData.infobox["area_urban_km2"]))
elif "area_blank2_km2" in infobox_keys: # En dernier recours ...
    typical_length = math.sqrt(float(pageData.infobox["area_blank2_km2"]))

print(typical_length)

# Récupération des coordonnées via Geopy

geolocator = Nominatim(user_agent="SatelliteExtractor")

location = geolocator.geocode(city)

# Conversion des coordonnées comme prévu

city_coords_wgs84 = (boundingBox(location.longitude,location.latitude, typical_length))

# Adaptation de la bounding box à la résolution imposée

resolution = (typical_length*1000)/2500 # première approximation

city_bbox = BBox(bbox=city_coords_wgs84, crs=CRS.WGS84)
city_size = bbox_to_dimensions(city_bbox, resolution=resolution)

if max(city_size) > 2500: # Si il y a un dépassement de taille -> on veut max 2500px en largeur ou hauteur
    resolution *= ((max(city_size))/2500) # Affinage
    city_size = bbox_to_dimensions(city_bbox, resolution=resolution)

print(resolution)

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
        input: ["DEM"],
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
    data_folder="C:/Users/migno/OneDrive/Bureau/satellite",
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A, # alternative : SENTINEL2_L1C
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

# Maintenant on récupère les données altimétriques
request_dem = SentinelHubRequest(
    data_folder="C:/Users/migno/OneDrive/Bureau/satellite",
    evalscript=evalscript_dem,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.DEM,
            time_interval=(prec_date, today),
            mosaicking_order=MosaickingOrder.LEAST_CC
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=city_bbox,
    size=city_size,
    config=config,
)

dem_imgs        = request_dem.get_data(save_data=True)

#image = true_color_imgs[0] # Pour récupérer l'image dans le script

print("Tâche finie \n")
