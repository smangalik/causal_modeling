# clear; python3.5 shootings_to_fips.py

import pandas as pd

mass_shootings_2019 = pd.read_csv("/data/smangalik/causal_modeling/2019_mass_shootings.csv")
mass_shootings_2020 = pd.read_csv("/data/smangalik/causal_modeling/2020_mass_shootings.csv")
city_to_fips = pd.read_csv("/data/smangalik/causal_modeling/uscities_top_population_90pct.csv")

# Stack the two mass shootings datasets
mass_shootings = pd.concat([mass_shootings_2019, mass_shootings_2020])
mass_shootings = mass_shootings.dropna(subset=['City Or County', 'State'])
mass_shootings = mass_shootings.rename(columns={'City Or County':'city', 'State':'state'})
mass_shootings['city'] = mass_shootings['city'].str.lower().str.strip()
mass_shootings['state'] = mass_shootings['state'].str.lower().str.strip()
print("\nMass Shootings Data", mass_shootings.shape)
print(mass_shootings.head())

# Relevant columns in the mass shootings dataset
city_to_fips = city_to_fips[['name','state_code','county','state','FIPS']]
city_to_fips = city_to_fips.rename(columns={'name':'city', 'state_code':'state_code', 'county':'county', 'state':'state', 'FIPS':'fips'})
city_to_fips['city'] = city_to_fips['city'].str.lower().str.strip()
city_to_fips['state'] = city_to_fips['state'].str.lower().str.strip()
city_to_fips['county'] = city_to_fips['county'].str.lower().str.strip()
city_to_fips['county'] = city_to_fips['county'].str.replace(' County', '').str.lower().str.strip()
city_to_fips['county'] = city_to_fips['county'].str.replace(' Parish', '').str.lower().str.strip()
city_to_fips['fips'] = city_to_fips['fips'].astype(str).str.zfill(5)
print("\nCity to FIPS Data", city_to_fips.shape)
print(city_to_fips.head())

# reset index
mass_shootings = mass_shootings.reset_index(drop=True)
city_to_fips = city_to_fips.reset_index()

def get_fips(city, state):
    # filter the city_to_fips dataset for state
    
    city_details = city.split('(')
    city = city_details[0]
    city = city.strip()
        
    county = None
    if len(city_details) > 1:
        county = city_details[1]
        county = county.replace(')','')
        
    # Adapting to the fips mapping data
    city = city.replace('saint','st')
    city = city.replace('winston salem','winston-salem')
    if state == 'alabama':
        city = city.replace('valhermoso springs','decatur')
    if state == 'ohio':
        city = city.replace('west chester','olde west chester')
    if state == 'mississippi':
        city = city.replace('cascilla','charleston')
        city = city.replace('hermanville','port gibson')
    if state == 'california':
        city = city.replace('sylmar','los angeles')
        city = city.replace('wilmington','los angeles')
        city = city.replace('canoga park','los angeles')
    if state == 'georgia':
        city = city.replace('lagrange','la grange')
    if state == 'tennessee':
        city = city.replace('beechgrove','tullahoma')
    if state == 'virginia':
        city = city.replace('fairfield','lexington')
    if state == 'west virginia':
        city = city.replace('williamsburg','lewisburg')
    
    city_to_fips_state = city_to_fips[city_to_fips['state'] == state]
    if len(city_to_fips_state) == 0:
        print("--- State NOT FOUND ---", state)
        return None
    # Try city matching
    city_to_fips_city = city_to_fips_state[city_to_fips_state['city'] == city]
    if len(city_to_fips_city) == 0:
        # Try county matching
        city_to_fips_city = city_to_fips_state[city_to_fips_state['county'] == county]
        if len(city_to_fips_city) == 0:
            # Try matching county with city
            city_to_fips_city = city_to_fips_state[city_to_fips_state['city'] == county]
            if len(city_to_fips_city) == 0:
                print("--- City NOT FOUND --- {} ({}) FROM {}".format(city,county,state))
                return None

    # return the fips code
    return city_to_fips_city['fips'].iloc[0]
    

# Apply the get_fips function to the mass shootings dataset
fips_matches = []
print("\nMatching Mass Shootings to FIPS")
for row in [row for _, row in mass_shootings.iterrows()]:
    city = row['city']
    state = row['state']
    fips = get_fips(city, state)
    fips_matches.append(fips)

print("\nMatched: ", len(fips_matches) - fips_matches.count(None), '/', len(fips_matches))

mass_shootings['fips'] = fips_matches
mass_shootings = mass_shootings.dropna(subset=['fips'])
print("\nMass Shootings Data with FIPS")
print(mass_shootings)
mass_shootings.to_csv("/data/smangalik/causal_modeling/mass_shootings_with_fips.csv", index=False)