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
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature

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

cursor.execute('''SELECT host_country, city, latitude, longitude, altitude FROM embassies WHERE country="'''+target_country+'''"''')
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


# Régions géographiques majeures (pour la disjonction selon l'albédo)
# Rappel pour GeoJSON : d'abord les longitudes puis les latitudes
# Afrique du Nord (Maghreb) + Asie de l'Ouest
magrheb_wasia = Polygon(
    [
        [
            (-17.714571587359274, 14.811771501389625),
            (-17.275118479932296, 21.896904120845846),
            (-14.462618592399574, 26.58334767025954),
            (-9.628634410702695, 32.76393006035829),
            (-4.135470567865348, 35.634733696968595),
            (1.9729276253697858, 36.8046471905232),
            (3.1044873829318678, 36.54123399797957),
            (4.302029094732827, 36.16868120573262),
            (5.060484155433843, 36.35791640953176),
            (6.416695839571907, 36.380347731757496),
            (7.637345527416676, 36.14877089850004),
            (9.252911290740634, 35.916508290273946),
            (10.222250748735014, 36.46701188470438),
            (10.940279976878992, 36.69764293500768),
            (28.94433099761968, 31.180652842974773),
            (36.10741664867959, 36.71756696264299),
            (43.27050229973949, 37.627972220575536),
            (44.58886162202046, 37.174140817551745),
            (50.47753326154209, 26.761101618853594),
            (56.27831427957834, 26.52542776581613),
            (62.43065778355617, 20.196006189196616),
            (45.2480412269273, 11.755287484946267),
            (40.76561953117201, 9.769467482169361),
            (38.612299304779775, 13.427868785636557),
            (36.63476032135833, 14.579108129147066),
            (32.89940890822893, 11.238526548233914),
            (36.63476032135833, 14.579108129147066),
            (32.89940890822893, 11.238526548233914),
            (30.877924614064774, 12.271080356968215),
            (29.295893427327627, 12.056286875628711),
            (28.768549411975126, 10.850344732422824),
            (22.537359708787477, 10.795169793971658),
            (21.656466211823496, 12.138065516150302),
            (15.627239610381155, 11.640024550144545),
            (8.912873622411277, 11.102674648064909),
            (4.15604840612058, 11.65919672654837),
            (-0.4833241951783923, 11.333091442633611),
            (-4.92694277449605, 12.214605635027635),
            (-7.784952961429955, 12.883373307809876),
            (-8.607120225263001, 14.367238091235093),
            (-11.954515513726122, 14.59468011617059),
            (-14.186112372701537, 13.949662216674165),
            (-15.732569845149412, 13.607444994332957),
            (-17.181150262379067, 13.702555527838074)
        ]
    ]
)
# Asie (hors Asie de l'Ouest)
easia = Polygon(
    [
        [
            (69.26164026478892, 22.503186499014717),
            (57.26059631155064, 25.34301350926836),
            (52.928079308436, 26.401625154307233),
            (49.03158232717756, 29.927407698410317),
            (47.28058098512172, 32.48429413178712),
            (45.571105425617716, 34.05812804418225),
            (44.034935838758344, 39.55824431734072),
            (48.10445658464131, 39.595585582374376),
            (49.88314176318899, 44.952167240868214),
            (49.8831417753663, 47.25974472191306),
            (63.80069672202273, 52.09052020236293),
            (70.16387889837031, 47.51696665914201),
            (73.34059664478035, 44.91736021954733),
            (77.09175362373496, 43.403146828455895),
            (81.90181651290123, 42.37380341129804),
            (85.5353172670946, 44.76031717806216),
            (86.71187940538863, 48.074876535498596),
            (90.70526969467075, 45.744474521089295),
            (110.77377980846364, 43.09331992578055),
            (108.41542117961002, 37.70445816026582),
            (106.06921785792439, 35.920433432925435),
            (92.34374089960733, 31.92839236523605),
            (81.19811533397304, 31.424195090091835),
            (74.74675675682761, 35.91588329591767),
            (71.45605502479398, 32.32685479030245),
            (75.5666050856663, 28.171874281206637)
        ]
    ]
)
# Australie
australia = Polygon(
    [
        [
            (111.95441663201959, -25.431204936865583),
            (113.58039312949944, -21.194998071690158),
            (120.08429911941884, -18.716844057489006),
            (122.1936740350684, -16.707584518441575),
            (127.42316601344956, -15.694796317450932),
            (128.03840036384733, -14.591913492893118),
            (131.37824398029244, -15.228894428624281),
            (132.60871268108804, -14.038365152520653),
            (138.40949369912425, -17.46368348427308),
            (140.5628139255165, -17.67316123378288),
            (142.40851697670985, -17.58939924202524),
            (144.2102747171605, -18.30012204591354),
            (147.8137901980618, -22.337713992229524),
            (146.27570432206738, -24.434981951875983),
            (147.7258995765764, -26.301131490452295),
            (150.01105573519675, -25.86696872383469),
            (150.27472759965292, -28.40863581669736),
            (147.46222771212024, -30.702012576789116),
            (147.9016808195472, -32.68361738932827),
            (146.05597776835384, -34.87496652995061),
            (143.5071497452773, -34.87496652995061),
            (141.52961076185588, -35.73566387528834),
            (138.32160307763885, -32.49848988766157),
            (136.9592984446152, -33.64008174738283),
            (134.32257980005326, -33.49361362743563),
            (128.43390816053164, -32.01537430046445),
            (123.81965053254827, -32.49848988766157),
            (123.20441618215047, -29.178864099309934),
            (117.53547109634233, -28.871456085003928),
            (119.86457256570539, -33.456957797381015),
            (114.45929934435341, -28.40863581669736)

]
    ]
)
# Corne de l'Afrique (Somalie, Ethiopie, etc.)
horn_of_africa = Polygon(
    [
        [
            (54.718175630658415, 12.951426896593528),
            (53.776431340003725, 9.702894592292752),
            (46.3471152692834, 1.9013890683610664),
            (42.51037927031986, 2.0059664859108755),
            (38.77828140592321, -4.233183199736996),
            (38.70852256957842, 0.7855719659376834),
            (36.19720446116591, 0.7158191199672663),
            (34.48811297071851, 4.095648950782732),
            (41.32447893250811, 4.964901669347465),
            (42.64989682305915, 9.496548557308415),
            (40.5920111508878, 9.462145416347603),
            (39.859543369267485, 12.30475267591733),
            (38.394607806026855, 11.963751884185696),
            (38.394607806026855, 13.528613030358533),
            (43.66139995005864, 12.475088000256076)
        ]
    ]
)
# Sud de l'Afrique
south_of_africa = Polygon(
    [
        [
            (9.585011278127492, -14.394326555659239),
            (13.144581448286095, -13.626849318623313),
            (13.715870487941181, -17.226935917089676),
            (30.722705745365626, -22.06739094482337),
            (25.311939608169972, -25.117509700395377),
            (27.24553328084872, -25.98972416854649),
            (28.16838487322285, -28.08368520482567),
            (26.366627097626168, -32.084506239245414),
            (24.883472796357907, -33.05662733825188),
            (22.411549067081104, -33.27734145425217),
            (19.51115855806298, -33.130260440663534),
            (16.303150647593448, -31.411857961697798),
            (9.634449458584744, -14.447527261158346)
        ]
    ]
)
# Ouest de l'Amérique du Sud (hors forêts)
west_of_samerica = Polygon(
    [
        [
            (-76.87189780139396, -14.447527398487024),
            (-75.90510096505459, -13.68022861165943),
            (-69.31330435364977, -14.830195412535833),
            (-64.83088265789449, -19.944512363252038),
            (-65.70978887274848, -24.0986997402135),
            (-67.15998412725754, -27.38354750119722),
            (-68.82990593548008, -30.460109408979623),
            (-71.55451520152742, -31.851501718967118)
        ]
    ]
)

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

    lat, lng, alt = embassy[2], embassy[3], embassy[4]
    city_coords_wgs84 = boundingBox(lat, lng, typical_length)

    # Adaptation de la bounding box à la résolution imposée

    resolution = (typical_length * 1000) / 2500  # première approximation

    city_bbox = BBox(bbox=city_coords_wgs84, crs=CRS.WGS84)
    city_size = bbox_to_dimensions(city_bbox, resolution=resolution)
    point = Feature(geometry=Point((lng, lat)))  # Ici, d'abord longitude puis latitude

    while max(
            city_size) > 2500:  # Tant qu'il y a un dépassement de taille ... (on veut max 2500px en largeur ou hauteur)
        resolution *= ((max(city_size)) / 2500)  # Affinage
        city_size = bbox_to_dimensions(city_bbox, resolution=resolution)

    # Requêtes d'acquisition d'images en couleurs "vraies"

    # Calcul du gain (le gain sert à éclaircir les images qui ont tendances à êtres beaucoup trop sombres)
    gain_color = 5.0  # gain par défaut (Europe)

    # (très lumineux, 0.4 < alb < 0.5 (été))
    if boolean_point_in_polygon(point, magrheb_wasia):
        gain_color = 2.75
    # (lumineux, 0.3 < alb < 0.4 (été))
    elif boolean_point_in_polygon(point, easia) \
            or boolean_point_in_polygon(point, australia) \
            or boolean_point_in_polygon(point, horn_of_africa) \
            or boolean_point_in_polygon(point, south_of_africa) \
            or boolean_point_in_polygon(point, west_of_samerica):
        gain_color = 3.3
    # (extrêmement lumineux, 0.9 < alb < 1.0 (été))
    elif embassy[0] == 'greenland':
        gain_color = 1.5

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

                gain = """ + str(gain_color) + """

                function evaluatePixel(sample) {
                    return [sample.B04 * gain, sample.B03 * gain, sample.B02 * gain];
                }
            """

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

                function evaluatePixel(sample) {
                  return [sample.DEM - """ + str(alt) + """]
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
                data_collection=DataCollection.SENTINEL2_L2A,
                # alternative : SENTINEL2_L1C (couleurs vraies, moins de contraste)
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

    print(str(round(count * 100 / total, 2)) + '%')
