from bs4 import BeautifulSoup
import urllib.request
import geopy
from geopy.geocoders import Nominatim
import ssl
import certifi

target_country = 'France'.lower() # Pays cible (en minuscule)

# Geopy
ctx = ssl._create_unverified_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
geolocator = Nominatim(timeout=10, scheme='https', user_agent="EmbassyExtractor") # timeout = 10 pour éviter de se faire déconnecter pendant les longues requêtes

embassies_home = urllib.request.urlopen("https://embassies.org/en/embassy-of-"+target_country)
embassies_home_bytes = embassies_home.read()

embassies_home_html = embassies_home_bytes.decode("utf8")
embassies_home.close()

soup_home = BeautifulSoup(embassies_home_html, 'html.parser') # Version scrappée de la page html

links_countries = soup_home.find_all('a') # Ensemble des liens de pays possèdant une ambassade française

embassies_list = []

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
            if href_j.find('https://www.google.com/maps/') != -1:  # Si le lien est un lien vers google maps c'est une adresse
                flag = True
                location = geolocator.geocode(links_embassy[j].get_text(), language='en')
                print(location)
            j += 1

        # CE QU'IL RESTE A FAIRE : TROUVER UNE CLE API GOOGLE GEOCODE POUR POUVOIR TROUVER DES GEOCODES POUR LES ADRESSES QUE GEOPY N'ARRIVE PAS A FAIRE

        #https://embassies.org/en/france-embassy-in-afghanistan
