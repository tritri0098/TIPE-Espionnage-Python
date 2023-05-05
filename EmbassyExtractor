import wptools
import geopy
from geopy.geocoders import Nominatim
import certifi
import ssl

ctx = ssl._create_unverified_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
geolocator = Nominatim(timeout=10, scheme='https', user_agent="SatelliteExtractor") # timeout = 10 pour éviter de se faire déconnecter pendant les longues requêtes

# Define the URL of the Wikipedia page to crawl
url = 'List_of_diplomatic_missions_of_France'

# Create a new wptools page object and fetch the page data
page = wptools.page(url).get_query()

# Get the table of diplomatic missions from the page data

# Loop through the rows of the table and extract the city names
cities = []

# Print the list of cities where an embassy of France is located
potential_cities = page.data['links']
for i in potential_cities:
    x = str(geolocator.geocode(i, language='en'))
    pos_comma = x.find(',') # S'il y a un virgule c'est que ce n'est pas le nom seul d'un pays mais bien une ville, une localité

    if x != None and pos_comma != -1: # Si x == None c'est que ce n'est pas un lieu (lien autre sur la page) et si len(x) == 1 C'est que c'est le nom d'un pays
        pos_par = x.find('(')  # S'il y a une parenthèse après le filtrage, ce n'est pas un nom de ville
        if pos_par == -1:
            print(x[:pos_comma])
            cities.append(x)

print(len(cities))
