from bs4 import BeautifulSoup
import urllib.request
import geopy
from geopy.geocoders import Nominatim, GoogleV3
import ssl
import certifi
import re
import unidecode # Permet de retirer l'accentuation
import mysql.connector
import requests
import time
import pandas as pd


target_country = 'Bahrain'.lower() # Pays cible (en minuscule)

# Geopy
ctx = ssl._create_unverified_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
# Geolocator : Nominatim
geolocator_nom = Nominatim(timeout=10, scheme='https', user_agent="EmbassyExtractor") # timeout = 10 pour éviter de se faire déconnecter pendant les longues requêtes
# Geolocator : GoogleV3
geolocator_google = GoogleV3(api_key="AIzaSyA0Br8sX9O4_zwtNGf35Zm5XFxboBvT0D8") # Attention l'API est payante : 0.005$ / requête : forfait gratuit 200$

# Requêtes pour obtenir les adresses des embassades
# Page d'accueil du site (uniquement le nom des pays cibles)
embassies_home = urllib.request.urlopen("https://embassies.org/en/embassy-of-"+target_country)
embassies_home_bytes = embassies_home.read()
# On récupère le code html de la page d'accueil
embassies_home_html = embassies_home_bytes.decode("utf8")
embassies_home.close()
# On scrap le site (version magnifiée du site)
soup_home = BeautifulSoup(embassies_home_html, 'html.parser') # Version scrappée de la page html

links_countries = soup_home.find_all('a') # Ensemble des liens de pays possèdant une ambassade française

embassies_list = [] # Liste des informations sur chaque ambassade. Format : [pays cible, pays hôte de l'ambassade, ville, adresse, lat, lng]

# Recherche des adresses des ambassades
for i in links_countries:
    href_country = i.get('href')
    if href_country.find(target_country+'-embassy-in-') != -1: # Si le lien est bien un lien vers une fiche d'ambassade

        embassies_emb = urllib.request.urlopen(href_country)
        embassies_emb_bytes = embassies_emb.read()
        embassies_emb_html = embassies_emb_bytes.decode("utf8")
        embassies_emb.close()

        soup_emb = BeautifulSoup(embassies_emb_html, 'html.parser') # Version scrappée de la page html

        links_embassy = soup_emb.find_all('a')  # Ensemble des liens d'embassades françaises
        j = 0
        flag = False
        while j < len(links_embassy) and flag == False:

            href_j = links_embassy[j].get('href')
            lat = 0.0 # Latitude  par défaut
            lng = 0.0 # Longitude par défaut
            alt = 0.0 # Altitude  par défaut
            alb = 0.5 # Albedo    par défaut (entre 0.0 et 1.0)
            city = "NullCity"       # Nom de la ville par défaut
            country = "NullCountry" # Nom du pays où se situe l'ambassade par défaut

            if href_j.find('https://www.google.com/maps/') != -1:  # Si le lien est un lien vers google maps c'est une adresse
                flag = True

                address = links_embassy[j].get_text().replace('\n                                            ', '').replace('    ', '') # Remplacer les retours à la ligne et espaces inutiles

                # Récupération des coordonnées GPS
                separator = 'query=' # Permet de séparer l'url des coordonnées GPS dans le lien
                len_sep = len(separator)
                pos_coords = href_j.find(separator)
                coords = href_j[pos_coords+len_sep:]
                regex_coords = r'(^[-+]?\d*\.?\d+),[ ]?([-+]?\d*\.?\d+)$' # Expression régulière permettant de détecter si coords est de la forme f,g avec f et g deux flottants. Les parenthèses permettent de séparer les deux coordonnées

                regex_match = re.match(regex_coords, coords) # On essaie de trouver des coordonnées GPS en clair dans l'url de la requête

                if regex_match == None: # Il n'y avait pas les coordonnées en clair dans l'url de la requête google maps
                    location_goo = geolocator_google.geocode(address, language='en') # On sort l'artillerie lourde
                    lat = location_goo.latitude
                    lng = location_goo.longitude

                    coords = str(lat)+','+str(lng) # On reconstitue la chaîne de caractère

                else:
                    # Obtention et conversions de la latitude / longitude
                    lat = float(regex_match.group(1)) # Latitude
                    lng = float(regex_match.group(2)) # Longitude

                try:
                    location_nom_by_coords = geolocator_nom.reverse(coords, language='en')  # Moins précis mais gratuit (suffisant pour le nom des villes, pays)
                except: # Les coordonnées ont été inversées sur le site
                    lng, lat = lat, lng
                    coords = (lat,lng)
                    location_nom_by_coords = geolocator_nom.reverse(coords, language='en')  # Moins précis mais gratuit (suffisant pour le nom des villes, pays)
                if location_nom_by_coords != None:
                    address_raw_nom = location_nom_by_coords.raw['address']
                    if address_raw_nom.get('city') is not None:
                        city = address_raw_nom['city']
                    elif address_raw_nom.get('town') is not None:
                        city = address_raw_nom['town']
                    elif address_raw_nom.get('municipality') is not None:
                        city = address_raw_nom['municipality']
                    elif address_raw_nom.get('village') is not None:
                        city = address_raw_nom['village']

                    if address_raw_nom.get('country'):
                        country = address_raw_nom['country']

                if city == 'NullCity' or country == 'NullCountry': # Si on a pas trouvé de ville ou si le nom de la ville est étrange ( parenthèses, etc. )
                    location_goo = geolocator_google.geocode(address, language='en')

                    address_raw_goo = location_goo.raw['address_components']
                    level3_flag = False
                    L = len(address_raw_goo)

                    if city == 'NullCity' or city.find('(') != -1 or city.find(')') != -1:
                        i = 0
                        while not level3_flag and i < L:
                            if address_raw_goo[i].get('types'):
                                if 'administrative_area_level_3' in address_raw_goo[i]['types'] or 'locality' in address_raw_goo[i]['types']:
                                    if address_raw_goo[i].get('short_name'):
                                        city = address_raw_goo[i]['short_name']
                                    elif address_raw_goo[i].get('long_name'):
                                        city = address_raw_goo[i]['long_name']
                                    level3_flag = True
                            i += 1
                    if country == 'NullCountry':
                        i = 0
                        while not level3_flag and i < L:
                            if address_raw_goo[i].get('types'):
                                if 'country' in address_raw_goo[i]['types']:
                                    if address_raw_goo[i].get('long_name'): # Pour les pays on préfère les noms longs (penser par exemple à united states of guam)
                                        city = address_raw_goo[i]['long_name']
                                    elif address_raw_goo[i].get('short_name'):
                                        city = address_raw_goo[i]['short_name']
                                    level3_flag = True
                            i += 1

                # Standardisation des formats
                country = country.lower()
                city = city.lower()
                address = unidecode.unidecode(address.lower())

                mun_detector = city.find('municipality') # permet de détecter les villes où il y aurait le mot 'municipality' qui ne fait pas partie du nom réel de la ville<
                if mun_detector != -1:
                    city = city[:mun_detector-1]

                # Détermination de l'altitude (élévation)
                query = ('https://api.opentopodata.org/v1/test-dataset?locations='+str(lat)+','+str(lng))
                try:
                    req = requests.get(query).json()
                    alt = pd.json_normalize(req, 'results').values[0][1]
                except:
                    () # On ne fait rien sinon

                time.sleep(0.5)  # Version gratuite de l'API OpenTopoData: il faut patienter entre chaque requête

                embassies_list.append([target_country, country, city, address, lat, lng, alt])

            j += 1

print("Données récupérées")


# Partie SQL

db = mysql.connector.connect(host="localhost", user="root", password="", database="embassies") # Création de la connexion avec la base de données
cursor = db.cursor()

print(embassies_list)

for i in range(len(embassies_list)):
    data = (embassies_list[i][0], embassies_list[i][1], embassies_list[i][2], embassies_list[i][3], embassies_list[i][4], embassies_list[i][5], embassies_list[i][6])
    cursor.execute('''INSERT INTO embassies (country, host_country, city, address, latitude, longitude, altitude) VALUES(%s, %s, %s, %s, %s, %s, %s)''', data)

db.commit() # Sauvegarde des changements sur la base de données

db.close() # Fin de la session de connexion à la base

print("Données sauvegardées dans la base de données")
