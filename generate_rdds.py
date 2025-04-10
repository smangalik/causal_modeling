"""
load the virtual environment: `source /data/smangalik/myenvs/diff_in_diff/bin/activate`
run as `python3.5 generate_rdds.py --covid_case`
run as `python3.5 generate_rdds.py --covid_death`
run as `python3.5 generate_rdds.py --worst_shooting`
run as `python3.5 generate_rdds.py --random`
"""

import matplotlib
matplotlib.use('Agg') 

import glob
import os
import sys  # noqa: F401
from typing import List, Tuple
from pymysql import cursors, connect # type: ignore
from collections import defaultdict
from sklearn.linear_model import LinearRegression

import warnings
import argparse
from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set numpy random seed
np.random.seed(0)

# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Process feature table for RDD analysis")
parser.add_argument('--covid_case', dest="covid_case", default=False ,action='store_true', help='Evaluate the first COVID-19 case per county')
parser.add_argument('--covid_death', dest="covid_death", default=False ,action='store_true', help='Evaluate the first COVID-19 death per county')
parser.add_argument('--worst_shooting', dest="worst_shooting", default=False ,action='store_true', help='Evaluate the worst fatal shooting per county')
parser.add_argument('--fires_start', dest="fires_start", default=False ,action='store_true', help='Evaluate when the CA fires started per county')
parser.add_argument('--mask_mandate', dest="mask_mandate", default=False ,action='store_true', help='Evaluate when the mask mandate started per county')
parser.add_argument('--outdoor_policy_start', dest="outdoor_policy_start", default=False ,action='store_true', help='Evaluate when the outdoor and recreation policy started per county')
parser.add_argument('--outdoor_policy_stop', dest="outdoor_policy_stop", default=False ,action='store_true', help='Evaluate when the outdoor and recreation policy stopped per county')
parser.add_argument('--childcare_policy_start', dest="childcare_policy_start", default=False ,action='store_true', help='Evaluate when the childcare policy started per county')
parser.add_argument('--entertainment_policy_start', dest="entertainment_policy_start", default=False ,action='store_true', help='Evaluate when the entertainment policy started per county')
parser.add_argument('--nonessential_business_policy_start', dest="nonessential_business_policy_start", default=False ,action='store_true', help='Evaluate when the nonessential business policy started per county')
parser.add_argument('--shelter_policy_start', dest="shelter_policy_start", default=False ,action='store_true', help='Evaluate when the shelter in place policy started per county')
parser.add_argument('--shelter_policy_stop', dest="shelter_policy_stop", default=False ,action='store_true', help='Evaluate when the shelter in place policy stopped per county')
parser.add_argument('--worship_policy_start', dest="worship_policy_start", default=False ,action='store_true', help='Evaluate when the worship policy started per county')
parser.add_argument('--restaurant_policy_start', dest="restaurant_policy_start", default=False ,action='store_true', help='Evaluate when the restaurant policy started per county')
parser.add_argument('--stay_home_policy_start', dest="stay_home_policy_start", default=False ,action='store_true', help='Evaluate when the mandatory stay at home policy started per county')
parser.add_argument('--advise_home_policy_start', dest="advise_home_policy_start", default=False ,action='store_true', help='Evaluate when the advise you stay at home policy started per county')
parser.add_argument('--carryout_policy_start', dest="carryout_policy_start", default=False ,action='store_true', help='Evaluate when the carryout policy started per county')
parser.add_argument('--social_distance_policy_start', dest="social_distance_policy_start", default=False ,action='store_true', help='Evaluate when the social distance policy started per county')
args = parser.parse_args()

# Where to load data from
data_file = "/data/smangalik/lbmha_yw_cnty.csv" # from research repo
#data_file = "/data/smangalik/lbmha_yw_cnty_undifferenced.csv" # from research repo

# How many of the top populous counties we want to keep
top_county_count = 4000 # Default to 4000, max is 3142

# user threshold (ut) required to consider a county
ut = 200 

# RDD Windows
default_before_start_window = 9 # additional weeks to consider before event start
default_after_end_window = 9    # additional weeks to consider after event end
event_buffer = 2                # number of weeks to ignore before and after event

# Confidence Interval Multiplier
ci_window = 1.96

# County-wise Socioeconomic Status
print("\nLoading SES Data...")
ses = pd.read_csv("/users2/smangalik/causal_modeling/LBMHA_Tract.csv")
ses['fips'] = ses['cfips'].astype(str).str.zfill(5)
ses = ses.groupby('fips').mean().reset_index()
ses = ses.dropna(subset=['fips','ses3'])
ses['ses3'] = ses['ses3'].astype(int)
print(ses.head(10))
county_to_ses = dict(zip(ses['fips'],ses['ses3']))

# event_date_dict[county] = [event_start (datetime, exclusive), [optional] event_end (datetime, inclusive), event_name]
county_events = {}

# populate events from countyFirsts.csv
first_covid_case = {}
first_covid_death = {}
fips_to_name = {}
fips_to_population = {}
with open("/data/smangalik/countyFirsts.csv") as countyFirsts:
    lines = countyFirsts.read().splitlines()[1:] # read and skip header
    for line in lines:
      fips, county, state, population, firstCase, firstDeath = line.split(",")
      fips_to_name[fips] = county + ", " + state
      fips_to_population[fips] = int(population)
      first_covid_case[fips] = [datetime.datetime.strptime(firstCase, '%Y-%m-%d'),None,"First Covid Case"]
      if firstDeath != "":
        first_covid_death[fips] = [datetime.datetime.strptime(firstDeath, '%Y-%m-%d'),None,"First Covid Death"]
        
# Load shooting events
shootings_df = pd.read_csv("/data/smangalik/causal_modeling/mass_shootings_with_fips.csv")
shootings_df['date'] = pd.to_datetime(shootings_df['Incident Date'])
shootings_df = shootings_df[shootings_df['date'].dt.year == 2020] # only keep shootings in 2020
shootings_df['fips'] = shootings_df['fips'].astype(str).str.zfill(5)
shootings_df['victims'] = shootings_df['Victims Killed'] + shootings_df['Victims Injured']
shootings_df['suspects'] = shootings_df['Suspects Killed'] + shootings_df['Suspects Injured']
# only keep the worst shooting for each county
shootings_df = shootings_df.sort_values('Victims Killed',ascending=False).drop_duplicates(subset='fips',keep='first').sort_index()
worst_shooting = {}
for i, row in shootings_df.iterrows():
  fips = row['fips']
  date = row['date']
  worst_shooting[fips] = [date, None, "Worst Shooting"]
  
# Load Wildfires Data
file_path = "/data/marif/california_fires_cleaned.csv"
print("\nLoading Wildfires Data from", file_path)
# Assuming cf_2020 is already filtered for 2020 incidents and has necessary columns
cf_2020 = pd.read_csv(file_path)
cf_2020 = cf_2020[pd.notna(cf_2020["fips_code"]) & ~cf_2020["fips_code"].isin([float('inf'), float('-inf')])]
cf_2020["fips_code"] = cf_2020["fips_code"].astype(int).astype(str).str.zfill(5)
county_fire_dict = {} # Initialize an empty dictionary to store the results
# Group by the 'fips_code' (or 'incident_county' if needed) to iterate over each county
for county, group in cf_2020.groupby('fips_code'):
    # Find the row with the maximum acres burned for the county
    max_fire = group.loc[group['incident_acres_burned'].idxmax()]
    # Extract the required values
    start_date = max_fire['incident_dateonly_created']
    event_name = max_fire['incident_name']
    # Add the information to the dictionary with the FIPS code as the key
    start_datetime = datetime.datetime.strptime(start_date, '%m/%d/%Y')
    county_fire_dict[county] = [start_datetime, None, event_name]
# print(county_fire_dict) # Display the resulting dictionary

# Load Mask Mandate Data
file_path = "/data/smangalik/mask_mandate.csv"
print("\nLoading mask mandate data from", file_path)
mask_mandate = pd.read_csv(file_path, usecols=["FIPS_Code", "Face_Masks_Required_in_Public", "date"])
#print("Filtering mask mandate data...")
mask_mandate = mask_mandate[pd.notna(mask_mandate["FIPS_Code"]) & ~mask_mandate["FIPS_Code"].isin([float('inf'), float('-inf')])]
mask_mandate["FIPS_Code"] = mask_mandate["FIPS_Code"].astype(int).astype(str).str.zfill(5)
# Filter rows where 'Face_Masks_Required_in_Public' is 'Yes'
df_yes = mask_mandate[mask_mandate["Face_Masks_Required_in_Public"] == "Yes"]
#print("Converting to Date Time...")
#mask_mandate.columns = mask_mandate.columns.str.strip()
df_yes["date"] = pd.to_datetime(df_yes["date"], format='%m/%d/%Y', errors='coerce')
# Sort data by FIPS_Code and date to ensure correct ordering
#print("Sorting mask mandate data...")
df_yes = df_yes.sort_values(by=["FIPS_Code", "date"])
mask_mandate = {}
# Identify the start and end dates for each county
#print("Identifying start and end dates for each county...")
for fips_code, group in df_yes.groupby("FIPS_Code"):
    start_date = group["date"].min()  # First occurrence of "Yes"
    end_date = group["date"].max()    # Last occurrence of "Yes"
    # if date not in 2020, skip
    if start_date.year != 2020:
      continue
    mask_mandate[fips_code] = [start_date, end_date, "mask_mandate"]
# print(mask_mandate)

# TODO Load policy events data
print("\nLoading policy events data...")
policy_events = {}
# policy_mapping = {
#   "/data/marif/restaurant_covid_policies.csv": ['Curbside/carryout/delivery only',
#                                                 'Open with social distancing/reduced seating/enhanced sanitation'],
#   "/data/marif/county_stay_at_home_orders.csv": ['Mandatory for all individuals','Advisory/Recommendation'],
# }
# fips = pd.read_csv("/data/marif/us_fips_codes.csv")
# fips["state_fips"] = fips["state_fips"].astype(str).str.zfill(2)
# fips["county_fips"] = fips["county_fips"].astype(str).str.zfill(3)
# states = pd.read_csv("/data/marif/state_abbreviations.csv")
# fips = fips.merge(states, on="state", how="left")
# fips["fips_code"] = fips["state_fips"].astype(str) + fips["county_fips"].astype(str)
# fips["county"] = fips["county"] + " County"
# for policy_csv, actions in policy_mapping.items():
#   for action in actions:
#     event_df = pd.read_csv(policy_csv, dtype={9: str, 15: str})  # Use str if they should be treated as strings
#     event_df = event_df.merge(fips[["county", "abbreviation", "fips_code"]],  
#                   left_on=["County_Name", "state"],  
#                   right_on=["county", "abbreviation"],  
#                   how="left")  
#     event_df["date"] = pd.to_datetime(event_df["date"], format="%m/%d/%Y", errors="coerce")
#     event_df = event_df.sort_values(by=["fips_code", "date"])
#     # Filter rows where Action is in the list of actions
#     try:
#       event_df = event_df[event_df['Action'] == action]
#     except Exception as e:
#       print("Error filtering actions: {} for {}".format(policy_csv, action))
#       print(e)
#       sys.exit(1)
#     # Group by fips_code and get the first state_date for each fips_code
#     policy_events[action] = {}
#     for fips_code, group in event_df.groupby('fips_code'):
#         # Get the first state_date for each fips_code
#         first_state_date = group['date'].iloc[0]
#         # Add to dictionary, with fips_code as the key and (state_date, None, 'mandatory') as the value
#         policy_events[action][fips_code] = (first_state_date, None, action)
  
# Load more county events
print("\nLoading more policy events data...")
events = "/data/marif/covid_county_policy_orders.csv"
policy_types = [("Food and Drink","start"),("Houses of Worship","start"),("Shelter in Place","start"),("Shelter in Place","stop"),
          ("Non-Essential Businesses","start"),("Entertainment","start"),("Childcare","start"),("Childcare","stop"),
          ("Outdoor and Recreation","start"),("Outdoor and Recreation","stop")]
for policy_type, start_stop in policy_types:
  event_df = pd.read_csv(events)
  df_county = event_df[event_df["policy_level"] == "county"]
  df_county = df_county[pd.notna(df_county["fips_code"]) & ~df_county["fips_code"].isin([float("inf"), float("-inf")])]
  df_county["fips_code"] = df_county["fips_code"].astype(int).astype(str).str.zfill(5)
  df_county = df_county[df_county["date"].astype(str).str[:4] == "2020"]
  df_county["date"] = pd.to_datetime(df_county["date"], format='%Y/%m/%d', errors='coerce')
  df_county = df_county.dropna(subset=["fips_code"])
  food_and_drink_start_dict = {
      row["fips_code"]: (row["date"], None, "{} {}".format(policy_type, start_stop.title()))
      for _, row in df_county[
          (df_county["policy_type"] == policy_type) & 
          (df_county["start_stop"] == start_stop)
      ].iterrows()
  }  
  
# Control events, randomly drawn from uniform distribution
control_events = {}
for fips in fips_to_population.keys():
  # select a random datetime.datetime in the year 2020
  random_date = datetime.datetime(2020, 1, 1) + datetime.timedelta(days=np.random.randint(0, 366))
  control_events[fips] = (random_date, None, "Control Event")
        
# Pick the events to use
if args.covid_case:
  print("\nUsing First Covid Case Events")
  county_events = first_covid_case
  event_name = "First Covid Case"
elif args.covid_death:
  print("\nUsing First Covid Death Events")
  county_events = first_covid_death
  event_name = "First Covid Death"
elif args.worst_shooting:
  print("\nUsing Worst Shooting Events")
  county_events = worst_shooting
  event_name = "Worst Shooting"
elif args.fires_start:
  print("\nUsing Wildfires Start Events")
  county_events = county_fire_dict
  event_name = "Wildfires Start"
elif args.mask_mandate:
  print("\nUsing Mask Mandate Events")
  county_events = mask_mandate
  event_name = "Mask Mandate"
else:
  print("\nNO EVENT CHOSEN, defaulting to control events")
  county_events = control_events
  event_name = "Control Event"

print('Connecting to MySQL...')

# Open default connection
connection  = connect(read_default_file="~/.my.cnf")

# Get the most populous counties
def get_populous_counties(cursor, base_year=2020) -> list:
  populous_counties = []
  sql = "select * from ctlb.counties{} order by num_of_users desc;".format(base_year)
  cursor.execute(sql)
  for result in cursor.fetchall():
    cnty, num_users = result
    cnty = str(cnty).zfill(5)
    populous_counties.append(cnty)
  return populous_counties

# Returns the yearweek that the date is within
def date_to_yearweek(d:datetime.datetime) -> str:
  if d is None:
    return None
  year, weeknumber, weekday = d.isocalendar()
  return str(year) + "_" + str(weeknumber).zfill(2)

def yearweek_to_dates(yw:str) -> Tuple[datetime.date, datetime.date]:
  '''
  Returns the first and last day of a week given as "2020_11"
  '''
  if yw is None:
    return None, None
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  return monday.date(), sunday.date()

def feat_usage_from_dates(county,dates,feat='Depression') -> float:
  feat_usages = []
  for date in sorted(dates):
    county_feats_county = county_feats.get(county,None)
    if county_feats_county is not None:
      county_feats_date = county_feats_county.get(date,None)
      if county_feats_date is not None: 
        county_feats_feat = county_feats_date.get(feat,None)
        if county_feats_feat is not None:
          feats_for_date = np.array(county_feats[county][date][feat])
          feat_usages.append(feats_for_date)
        else:
          #print("No feature",feat,"for",county,"on",date)
          feat_usages.append(np.nan)
      else:
        #print("No date",date,"for",county)
        feat_usages.append(np.nan)
    else:
      #print("No county",county)
      return [np.nan] * len(dates)
      
  return feat_usages

def regression_feat_usage_from_dates(county,dates,feat='Depression') -> float:
  # Get the feature usage for the county on the dates
  feat_usages = feat_usage_from_dates(county,dates,feat)
  
  # If all X is NaN, return NaN
  if all(np.isnan(feat_usages)):
    return np.nan, np.nan

  # Report the slope and intercept of the date and feat usage
  feat_usages = np.array(feat_usages)
  mask = ~np.isnan(feat_usages) # drop NaNs
  X = np.arange(len(feat_usages))[mask].reshape(-1, 1)
  y = feat_usages[mask]
  
  regression = LinearRegression().fit(X, y)
  return regression.coef_[0], regression.intercept_


def regression_usage_before_and_after(county, feat, event_start, event_end=None,
                                 before_start_window=default_before_start_window,
                                 after_start_window=default_after_end_window,
                                 event_buffer=0) -> Tuple[ Tuple[float,float], Tuple[float,float], List[str], List[str] ]:
  '''
  Returns the average feature usage for a county before and after an event
  @param county: the county to examine
  @param feat: the feature to examine
  @param event_start: the start of the event
  @param event_end: the end of the event
  @param before_start_window: the number of weeks before the event to consider
  @param after_start_window: the number of weeks after the event to consider
  @param event_buffer: the number of weeks to ignore before and after the event
  @return: the average feature usage before and after the event as well as the dates considered
  '''

  # If no event end specified, end = start
  if event_end is None:
    event_end = event_start

  #print('center',date_to_yearweek(event_start))

  # Apply buffer
  event_start = event_start - datetime.timedelta(days=event_buffer*7)
  event_end = event_end + datetime.timedelta(days=event_buffer*7)

  #print('start',date_to_yearweek(event_start))
  #print('end',date_to_yearweek(event_end))

  # Before window dates, ex. '2020_11',2 -> ['2020_08', '2020_09', '2020_10']
  before_dates = []
  for i in range(before_start_window + 1):
    day = event_start - datetime.timedelta(days=i*7)
    before_dates.append(day)
  before_dates = [date_to_yearweek(x) for x in before_dates]
  before_dates = list(set(before_dates))
  before_dates.sort()
  #print('before',before_dates)

  # After window dates, ex. '2020_11',2 -> ['2020_11', '2020_12', '2020_13']
  after_dates = []
  for i in range(after_start_window + 1):
    day = event_end + datetime.timedelta(days=i*7)
    after_dates.append(day)
  after_dates = [date_to_yearweek(x) for x in after_dates]
  after_dates = list(set(after_dates))
  after_dates.sort()
  #print('after',after_dates)

  # Get average usage
  return regression_feat_usage_from_dates(county, before_dates, feat), regression_feat_usage_from_dates(county, after_dates, feat), before_dates, after_dates

# Read in data with cursor
with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    # Determine the relevant counties
    #print("\nCounties with the most users in {}".format(base_year),populous_counties[:25],"...")
    populous_counties = sorted(fips_to_population, key=fips_to_population.get, reverse=True)[:top_county_count]
    print("\nCounties with the most users in 2021",populous_counties[:25],"...")

    # Load data from CSV
    df = pd.read_csv(data_file)
    df['cnty'] = df['cnty'].astype(str).str.zfill(5)
    df['feat'] = df['score_category'].str.replace('DEP_SCORE','Depression').replace('ANX_SCORE','Anxiety').replace('WEC_sadF','Sad').replace('WEB_worryF','Worry')
    score_col = 'score'
    std_col = 'score_std'

    # Require n_users >= UT for each county
    df = df[df['n_users'] >= ut]

    print("Data from CSV:", data_file)
    print(df.head(10))
    print("Columns:",df.columns.tolist())

    list_features = df['feat'].unique()
    
    # Normalize feat scores per county feature
    #df[score_col] = df.groupby(['cnty','feat'])[score_col].transform(lambda x: (x - x.min()) / (x.max() - x.min())) # Min Max Scale
    df[score_col] = df.groupby(['cnty','feat'])[score_col].transform(lambda x: (x - x.mean()) / x.std()) # Z-Score 
    
    print("Loading Unique Features:",list_features)
    county_feats = defaultdict(lambda: defaultdict(dict))
    county_feats_std = defaultdict(lambda: defaultdict(dict))
    grouped = df.groupby(['feat', 'cnty', 'yearweek'])
    for (feat, county, yearweek), group in tqdm(grouped):
        county_feats[county][yearweek][feat] = group[score_col].iloc[0]
        county_feats_std[county][yearweek][feat] = group[std_col].iloc[0]
    county_feats = {k: dict(v) for k, v in county_feats.items()} # Convert to a regular dict

    # Calculate Average and Weighted Average Weekly Feat Score for Baselines
    avg_county_list_usages = {} # avg_county_list_usages[yearweek][feat] = average value
    all_possible_yearweeks = sorted(df['yearweek'].unique().tolist())
    for yearweek in all_possible_yearweeks:
      for feat in list_features:
        feat_usages = []
        for county in set(populous_counties):
          if county_feats.get(county):
            if county_feats[county].get(yearweek): 
              if county_feats[county][yearweek].get(feat):
                feats_for_date = county_feats[county][yearweek][feat]
                feat_usages.append(feats_for_date)
        if not feat_usages:
          # print("No matching dates for", county, "on dates", dates)
          continue
        avg_county_list_usages[yearweek] = avg_county_list_usages.get(yearweek,{})
        avg_county_list_usages[yearweek][feat] = round( np.nanmean(feat_usages), 5)
        avg_county_list_usages[yearweek][feat+'_unique'] = round( len(set(feat_usages)), 5)
        avg_county_list_usages[yearweek][feat+'_std'] = round( np.nanstd(feat_usages), 5)
        
    num_counties_studied = len(set(populous_counties).intersection(set(county_feats.keys())))
    print("\nAverage Feature Usage for", num_counties_studied, "studied counties")
    for yearweek in all_possible_yearweeks:
      print(yearweek,avg_county_list_usages[yearweek])
  
    # Calculate RDDs dict[county] = value
    target_befores = {}
    target_afters = {}
    target_diffs = {}

    output_dicts = []

    # Delete all previous plots
    print("Deleting all previous plots...")
    files = glob.glob('/users2/smangalik/causal_modeling/rdd_plots/*')
    for f in files:
      os.remove(f)
    
    target_counties = set(populous_counties)
    missing_counties = set()
    x_center = default_before_start_window + 1
    
    # All outcomes
    x_before_all = {feat:[] for feat in list_features}
    y_before_all = {feat:[] for feat in list_features}
    x_after_all = {feat:[] for feat in list_features}
    y_after_all = {feat:[] for feat in list_features}
    x_all = {feat:[] for feat in list_features}
    y_all = {feat:[] for feat in list_features}
    
    # Spaghetti outcomes
    x_before_spaghetti = {feat:[] for feat in list_features}
    y_before_spaghetti = {feat:[] for feat in list_features}
    x_after_spaghetti = {feat:[] for feat in list_features}
    y_after_spaghetti = {feat:[] for feat in list_features}
    x_spaghetti = {feat:[] for feat in list_features}
    y_spaghetti = {feat:[] for feat in list_features}
    
    # SES outcomes
    x_before_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    y_before_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    x_after_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    y_after_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    x_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    y_ses = {feat:{1:[], 2:[], 3:[]} for feat in list_features}
    
    # County outcomes
    x_before_county = {feat:{} for feat in list_features}
    y_before_county = {feat:{} for feat in list_features}
    x_after_county = {feat:{} for feat in list_features}
    y_after_county = {feat:{} for feat in list_features}
    x_county = {feat:{} for feat in list_features}
    y_county = {feat:{} for feat in list_features}
    
    print("\nCalculating RDDs for",len(target_counties),"counties")
    for target in tqdm(target_counties):
      
      if target not in county_feats:
        continue # no data for this county
      
      # Event for each county
      blank_event = [None,None,None]
      target_event_start, target_event_end, _ = county_events.get(target,blank_event)

      # If no event was found for this county, skip
      if target_event_start is None and target_event_end is None:
        #print("Skipping {} ({}) due to missing event".format(target,fips_to_name.get(target)))
        missing_counties.add(target)
        continue # no event was found for this county
      
      for feat in list_features: # run RDD against each feature
        
        (target_before_slope,target_before_intercept), (target_after_slope,target_after_intercept), dates_before, dates_after = regression_usage_before_and_after(target, feat, event_start=target_event_start, event_end=target_event_end)
        if not all([target_before_slope,target_after_slope]):
          #print("Skipping {} ({}) for {} due to missing data".format(target,fips_to_name.get(target),feat))
          continue
        
        # DEBUG, check if the dates are correct
        # print("target",target, fips_to_name.get(target))
        # print("target_event_start",target_event_start, date_to_yearweek(target_event_start))
        # print("date_before",dates_before)
        # print("date_after",dates_after)
        # sys.exit()

        # How the target county changed
        target_before = (target_before_slope, target_before_intercept)
        target_after = (target_after_slope, target_after_intercept)
        target_diff = (target_after_slope - target_before_slope, target_after_intercept - target_before_intercept)
        target_befores[target] = target_before
        target_afters[target] = target_after
        target_diffs[target] = target_diff
        
        # RDD Calculation
        slope_effect, intercept_effect = target_diff
        is_significant = abs(intercept_effect) >= 0.2

        # print("County: {} ({}) studying {} against {}".format(target, fips_to_name.get(target), feat, event_name))
        
        increase_decrease = "increased" if intercept_effect > 0 else "decreased"
        
        # Relevant Dates
        begin_before, _ = yearweek_to_dates(min(dates_before))
        _, end_before = yearweek_to_dates(max(dates_before))
        begin_after, _ = yearweek_to_dates(min(dates_after))
        _, end_after = yearweek_to_dates(max(dates_after))
        middle_before = begin_before + (end_before - begin_before)/2
        middle_after = begin_after + (end_after - begin_after)/2
        middle_middle = middle_before + (middle_after - middle_before)/2 

        # --- Calculate Before and After Values ---

        # Calculate in-between dates and xticks
        x = np.array(range(1, x_center + len(dates_after)))
        xticklabels = [
            x_i - x_center for x_i in x
          ]
        xticklabels[x_center-1] = target_event_start.strftime('%Y-%m-%d')
        x_before = np.array(x[:x_center])
        x_after = np.array(x[x_center-1:])
        
        # Drop NaNs
        # y_before = [ avg_county_list_usages[yw][feat] if yw in avg_county_list_usages else np.nan for yw in dates_before ]
        # y_after =  [ avg_county_list_usages[yw][feat] if yw in avg_county_list_usages else np.nan for yw in dates_after ]
        y_before = [ county_feats[target][yw][feat] if yw in county_feats[target].keys() else np.nan for yw in dates_before ]
        y_after =  [ county_feats[target][yw][feat] if yw in county_feats[target].keys() else np.nan for yw in dates_after ]
        y_before_std = [ county_feats_std[target][yw][feat] if yw in county_feats_std[target].keys() else 0.0 for yw in dates_before ]
        y_after_std =  [ county_feats_std[target][yw][feat] if yw in county_feats_std[target].keys() else 0.0 for yw in dates_after ]
        y_all_std = y_before_std + y_after_std[1:] # skip the first repeat date in "after"
        y_vals = np.array(y_before + y_after[1:]) # skip the first repeat date in "after"
        y_before = np.array(y_before)
        y_after = np.array(y_after)
        # First after that is not NaN - Last before that is not NaN
        target_discontinuity = y_after[event_buffer] - y_before[-event_buffer]
        
        # Arrays to list for saving (gives a meaningful speed up)
        x_before_list = x_before.tolist()
        x_after_list = x_after.tolist()
        x_list = list(np.concatenate([x_before,x_after[1:]]))
        y_before_list = y_before.tolist()
        y_after_list = y_after.tolist()
        y_list = list(np.concatenate([y_before,y_after[1:]]))
        
        # Store all y_vals for mega-plot
        x_before_all[feat].extend(x_before_list)
        x_after_all[feat].extend(x_after_list)
        x_all[feat] += x_list
        y_before_all[feat].extend(y_before_list)
        y_after_all[feat].extend(y_after_list)
        y_all[feat] += y_list
        
        # Store all vals for spaghetti-plot
        x_before_spaghetti[feat].append(x_before_list)
        x_after_spaghetti[feat].append(x_after_list)
        x_spaghetti[feat].append(x_list)
        y_before_spaghetti[feat].append(y_before_list)
        y_after_spaghetti[feat].append(y_after_list)
        y_spaghetti[feat].append(y_list)
        
        # Store all vals for SES-plot
        x_before_ses[feat][county_to_ses[target]].append(x_before_list)
        x_after_ses[feat][county_to_ses[target]].append(x_after_list)
        x_ses[feat][county_to_ses[target]].append(x_list)
        y_before_ses[feat][county_to_ses[target]].append(y_before_list)
        y_after_ses[feat][county_to_ses[target]].append(y_after_list)
        y_ses[feat][county_to_ses[target]].append(y_list)
        
        # Store all vals for County-plot
        # Skip if empty
        if x_before.size > 0 and x_after.size > 0:
          x_before_county[feat][target] = x_before_list
          x_after_county[feat][target] = x_after_list
          x_county[feat][target] = list(x_list)
          y_before_county[feat][target] = y_before_list
          y_after_county[feat][target] = y_after_list
          y_county[feat][target] = y_list
        
        
        # For writing to a dataframe
        pandas_entry = {
          "fips": target, # FIPS code of the county
          "county": fips_to_name.get(target), # Name of the county
          "feat": feat, # Either Depression or Anxiety
          "event_name": event_name, # Name of the event that we are measuring the feat before and after
          "event_date": target_event_start, # Date of the event
          "event_date_centered": middle_middle, # The date in the middle of the week containing the event_date   
          "target_before": str(y_before), # feat scores before the event
          "target_after": str(y_after), # feat scores after the event
          "target_before_slope": target_before[0], # feat score before the event
          "target_after_slope": target_after[0], # feat score after the event
          "target_before_intercept": target_before[1], # feat score before the event
          "target_after_intercept": target_after[1], # feat score after the event
          "target_diff_slope": target_diff[0], # feat score after - feat score before
          "target_diff_intercept": target_diff[1], # feat score after - feat score before
          "target_discontinuity": target_discontinuity, # feat score after - feat score before (w/ event_buffer)
          "slope_effect": slope_effect,
          "intercept_effect": intercept_effect, 
          "ses": county_to_ses.get(target), # Socioeconomic Status of the county
        }
        output_dicts.append(pandas_entry)   
        
        # Only plot significant results
        if not is_significant:
          continue # only plot significant results    
        
        # --- Plotting ---

        # Create County-Level RDD Plot
        plt.clf() # reset old plot
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # Horizontally stacked plots
        fig.set_size_inches(20, 6)
        ax1.scatter(x[:x_center-1], y_vals[:x_center-1], label='Target (Before)')
        ax1.scatter(x[x_center-1], y_vals[x_center-1], color='black',label='Target (During)')
        ax1.scatter(x[x_center:], y_vals[x_center:], label='Target (After)')       
        # Add error bars
        ax1.errorbar(x_before, y_before, yerr=y_before_std, fmt='o', capsize=4.0, color='blue',  alpha=0.3)
        ax1.errorbar(x_after, y_after, yerr=y_after_std, fmt='o', capsize=4.0, color='orange', alpha=0.3)
        
        # Plot vertical line for event
        ax1.axvline(x=x_center, color='black', linestyle='--', label='Event')
        ax2.axvline(x=x_center, color='black', linestyle='--', label='Event')
        
        # Plot line of best fit for Target (Before)
        mask = ~np.isnan(y_before)
        regression = LinearRegression().fit( x_before[mask].reshape(-1, 1), y_before[mask] )
        ax1.plot(x_before, regression.predict(x_before.reshape(-1, 1)), color='blue',linestyle='--', alpha=0.3, label='Target (Before) Fit')
        
        # Plot line of best fit for Target (After)
        mask = ~np.isnan(y_after)
        regression = LinearRegression().fit( x_after[mask].reshape(-1, 1), y_after[mask] )
        ax1.plot(x_after, regression.predict(x_after.reshape(-1, 1)), color='orange', linestyle='--', alpha=0.3, label='Target (After) Fit')
        
        
        # Plot Line of Best Fit for All
        mask = ~np.isnan(y_vals)
        regression = LinearRegression().fit( x[mask].reshape(-1, 1), y_vals[mask] )
        ax2.plot(x, regression.predict(x.reshape(-1, 1)), color='green', linestyle='--', alpha=0.3, label='All Fit')
        
        # Set x label for ax1
        ax1.set_xlabel("Weeks Before and After Event")
        ax2.set_xlabel("Weeks Before and After Event")
        
        fig.suptitle('Effect of {} on {} (Slope: {} Intercept: {})'.format(
          event_name, fips_to_name.get(target), round(target_diff[0],3), round(target_diff[1],3))
        )
        
        # Plot the average and weighted average per week
        ax2.scatter(x, y_vals, color='green', label='All', alpha=0.8)
        ax2.errorbar(x, y_vals, yerr=y_all_std, fmt='o', color='green', capsize=4.0, alpha=0.3)


        ymin = min([min([y for y in y_vals if y is not None])])
        ymax = max([max([y for y in y_vals if y is not None])])

        # Format plot
        plt.ylabel("County {} Z-Score".format(feat))
        for ax in [ax1, ax2]:
          ax.set_xticks(x)
          ax.set_xticklabels(xticklabels, rotation=0)
          ax.axis(ymin=ymin, ymax=ymax)
          ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt_name = "rdd_plots/rdd_{}_{}_before_after_event.png".format(target, feat)

        plt.savefig(plt_name) # optionally save all figures
        
    print("\n",len(missing_counties),"counties with no {} dates:\n".format(event_name),missing_counties)
        
    # Create a mega-plot of all befores and afters     
    for feat in list_features:
      print("\nCreating Mega-Plot for",feat)
      x_before_all_feat = np.array(x_before_all[feat])
      y_before_all_feat = np.array(y_before_all[feat])
      x_after_all_feat = np.array(x_after_all[feat])
      y_after_all_feat = np.array(y_after_all[feat])
      x_during_all_feat = np.array(x_all[feat])
      y_during_all_feat = np.array(y_all[feat])
      x_all_feat = np.array(x_all[feat])
      y_all_feat = np.array(y_all[feat])
      plt.clf() # reset old plot
      x = np.array(range(1, default_before_start_window + 1 + default_before_start_window + 1))
      xticklabels = [x_i - x_center for x_i in x]
      xticklabels[x_center-1] = ""
      circle_size = 100
      circle_alpha = 0.3
      line_alpha = 0.5
      fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # Horizontally stacked plots
      fig.set_size_inches(20, 6)
      ax1.scatter(x_before_all_feat, y_before_all_feat, label='Target (Before)', edgecolor='red', facecolors='none', alpha=circle_alpha)
      ax1.scatter(x_after_all_feat, y_after_all_feat, label='Target (After)', edgecolor='blue', facecolors='none', alpha=circle_alpha)
      # Plot vertical line for event
      ax1.axvline(x=x_center, color='black', linestyle='--', label='Event')
      ax2.axvline(x=x_center, color='black', linestyle='--', label='Event')
      # Plot line of best fit for Target (Before)
      mask = ~np.isnan(y_before_all_feat)
      regression_before = LinearRegression().fit( x_before_all_feat[mask].reshape(-1, 1), y_before_all_feat[mask] )
      ax1.plot(x_before_all_feat, regression_before.predict(x_before_all_feat.reshape(-1, 1)), color='red', linestyle='--', alpha=line_alpha, label='Target (Before) Fit')
      # Plot line of best fit for Target (After)
      mask = ~np.isnan(y_after_all_feat)
      try:
        regression_after = LinearRegression().fit( x_after_all_feat[mask].reshape(-1, 1), y_after_all_feat[mask] )
        ax1.plot(x_after_all_feat, regression_after.predict(x_after_all_feat.reshape(-1, 1)), color='blue', linestyle='--', alpha=line_alpha, label='Target (After) Fit')
      except Exception as e:
        print("Error with plotting",feat)
        print(e)
        continue
      # Plot Line of Best Fit for All
      mask = ~np.isnan(y_all_feat)
      regression_all = LinearRegression().fit( x_all_feat[mask].reshape(-1, 1), y_all_feat[mask] )
      ax2.plot(x_all_feat, regression_all.predict(x_all_feat.reshape(-1, 1)), color='green', linestyle='--', alpha=line_alpha, label='All Fit')
      # Set x label for ax1
      ax1.set_xlabel("Weeks Before and After Event")
      ax2.set_xlabel("Weeks Before and After Event")
      slope_before = regression_before.coef_[0]
      slope_after = regression_after.coef_[0]
      intercept_before = regression_before.intercept_
      intercept_after = regression_after.intercept_
      fig.suptitle('Effect of {} (Slope: {} Intercept: {})'.format(
        event_name, round(slope_after-slope_before,3), round(intercept_after-intercept_before,3))
      )
      # Plot the average and weighted average per week
      ax2.scatter(x_all[feat], y_all[feat], label='All', edgecolor='green', facecolors='none', alpha=circle_alpha)
      ymin = min([min([y for y in y_all_feat if y is not None])]) - 0.05
      ymax = max([max([y for y in y_all_feat if y is not None])]) + 0.05
      # Format plot
      plt.ylabel("{} Z-Score".format(feat))
      for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.axis(ymin=ymin, ymax=ymax)
        ax.legend()
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt_name = "rdd_plots_all/rdd_{}_{}_before_after_{}.png".format(
        num_counties_studied,feat.lower(),event_name.lower().replace(" ","_")
      )
      plt.savefig(plt_name) # optionally save all figures
    
    # Create a spaghetti plot of all befores and afters     
    for feat in list_features:
      print("\nCreating Spaghetti-Plot for",feat)
      x_before_spaghetti_feat = np.array(x_before_spaghetti[feat])
      y_before_spaghetti_feat = np.array(y_before_spaghetti[feat])
      x_after_spaghetti_feat = np.array(x_after_spaghetti[feat])
      y_after_spaghetti_feat = np.array(y_after_spaghetti[feat])
      x_spaghetti_feat = np.array(x_spaghetti[feat])
      y_spaghetti_feat = np.array(y_spaghetti[feat])
      # Create a buffer around event_buffer
      if event_buffer > 0:
        # Remove the last event_buffer weeks from the end of before
        x_before_spaghetti_feat = x_before_spaghetti_feat[:,:-event_buffer]
        y_before_spaghetti_feat = y_before_spaghetti_feat[:,:-event_buffer]
        # Remove the first event_buffer weeks from the start of after
        x_after_spaghetti_feat = x_after_spaghetti_feat[:,event_buffer:]
        y_after_spaghetti_feat = y_after_spaghetti_feat[:,event_buffer:]
        # Concatenate the removed weeks into a during
        x_during_spaghetti_feat = np.concatenate([x_before_spaghetti_feat[:,-event_buffer+1:],x_after_spaghetti_feat[:,:event_buffer-1]],axis=1)
        y_during_spaghetti_feat = np.concatenate([y_before_spaghetti_feat[:,-event_buffer+1:],y_after_spaghetti_feat[:,:event_buffer-1]],axis=1)
      plt.clf() # reset old plot
      x = np.array(range(1, x_center + len(dates_after)))
      x_before = x_before_spaghetti_feat[0]
      x_after = x_after_spaghetti_feat[0]
      xticklabels = [x_i - x_center for x_i in x]
      xticklabels[x_center-1] = ""
      line_alpha = 0.05
      avg_alpha = 0.5
      fill_alpha = 0.3
      fit_alpha = 0.5
      fig, ax1 = plt.subplots(nrows=1, ncols=1) # Horizontally stacked plots
      fig.set_size_inches(8, 6)
      # # Spaghetti (very thin) Plots
      # for x_line, y_line in zip(x_before_spaghetti_feat, y_before_spaghetti_feat):
      #   ax1.plot(x_line, y_line, color='red', alpha=line_alpha)
      # for x_line, y_line in zip(x_after_spaghetti_feat, y_after_spaghetti_feat):
      #   ax1.plot(x_line, y_line, color='blue', alpha=line_alpha)
      # for x_line, y_line in zip(x_spaghetti_feat, y_spaghetti_feat):
      #   ax2.plot(x_line, y_line, color='green', alpha=line_alpha)
      # Standard Deviation Shading
      ax1.fill_between(x_before_spaghetti_feat[0],
                       np.nanmean(y_before_spaghetti_feat, axis=0) - np.nanstd(y_before_spaghetti_feat, axis=0),
                       np.nanmean(y_before_spaghetti_feat, axis=0) + np.nanstd(y_before_spaghetti_feat, axis=0),
                       color='red', alpha=fill_alpha)
      ax1.fill_between(x_after_spaghetti_feat[0],
                       np.nanmean(y_after_spaghetti_feat, axis=0) - np.nanstd(y_after_spaghetti_feat, axis=0),
                       np.nanmean(y_after_spaghetti_feat, axis=0) + np.nanstd(y_after_spaghetti_feat, axis=0),
                       color='blue', alpha=fill_alpha)
      # ax2.fill_between(x, 
      #                  np.nanmean(y_spaghetti_feat, axis=0) - np.nanstd(y_spaghetti_feat, axis=0), 
      #                  np.nanmean(y_spaghetti_feat, axis=0) + np.nanstd(y_spaghetti_feat, axis=0), 
      #                  color='green', alpha=fill_alpha)
      # Plot average of Target (Before), Target (After), and All
      ax1.plot(x_before, np.nanmean(y_before_spaghetti_feat, axis=0), color='red', linestyle='--', alpha=avg_alpha, label='Average (Before)')
      ax1.plot(x_after, np.nanmean(y_after_spaghetti_feat, axis=0), color='blue', linestyle='--', alpha=avg_alpha, label='Average (After)')
      if event_buffer > 0:
        x_during = x[len(x_before)-1:-len(x_after)+1]
        y_during = np.nanmean(y_spaghetti_feat, axis=0)[len(x_before)-1:-len(x_after)+1]
        ax1.plot(x_during, y_during, color='green', linestyle='--', alpha=avg_alpha, label='Average (During)')
      #ax2.plot(x, np.nanmean(y_spaghetti_feat, axis=0), color='green', linestyle='--', alpha=1, label='Average')
      # Linear Regression Line Before, After, All
      try:
        regression_before = LinearRegression().fit( x_before.reshape(-1, 1), np.nanmean(y_before_spaghetti_feat, axis=0) )
        ax1.plot(x_before, regression_before.predict(x_before.reshape(-1, 1)), color='red', label='Fit (Before)', alpha=fit_alpha)
        regression_after = LinearRegression().fit( x_after.reshape(-1, 1), np.nanmean(y_after_spaghetti_feat, axis=0) )
        ax1.plot(x_after, regression_after.predict(x_after.reshape(-1, 1)), color='blue', label='Fit (After)', alpha=fit_alpha)
      except Exception as e:
        print("Error with plotting",feat)
        print(e)
        continue
      regression_all = LinearRegression().fit( x.reshape(-1, 1), np.nanmean(y_spaghetti_feat, axis=0) )
      #ax2.plot(x, regression_all.predict(x.reshape(-1, 1)), color='green', label='Fit')
      # Plot vertical line for event
      ax1.axvline(x=x_center, color='black', linestyle='--')
      #ax2.axvline(x=x_center, color='black', linestyle='--')
      # Set x label for ax1
      ax1.set_xlabel("Weeks Before and After {}".format(event_name))
      #ax2.set_xlabel("Weeks Before and After {}".format(event_name))
      slope_change = regression_after.coef_[0] - regression_before.coef_[0]
      intercept_change = regression_after.intercept_ - regression_before.intercept_
      y_change = np.nanmean(y_after_spaghetti_feat, axis=0)[0] - np.nanmean(y_before_spaghetti_feat, axis=0)[-1]
      fig.suptitle('Effect of {} (Slope: {}, Intercept: {}, Discontinuity: {})'.format(
        event_name, '{:+}'.format(round(slope_change,3)), '{:+}'.format(round(intercept_change,3)), '{:+}'.format(round(y_change,3)))
      )
      ymin = np.nanmin(y_spaghetti_feat) - 0.05
      ymax = np.nanmax(y_spaghetti_feat) + 0.05
      # Format plot
      plt.ylabel("{} Z-Score".format(feat))
      for ax in [ax1]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.axis(ymin=ymin, ymax=ymax)
        ax.legend()
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt_name = "rdd_plots_all/rdd_{}_{}_spaghetti_{}.png".format(
        num_counties_studied, feat.lower(), event_name.lower().replace(" ","_")
      )
      plt.savefig(filename=plt_name)   
      
    def plot_by_subsection(subsections:dict, subsection_str:str="", colors:list=[]):
      '''
      subsections: dict of subsection_names and their counties (as fips codes)
      colors: list of colors for each subsection
      filename_str: string to append to the filename
      '''
      if not subsections:
        print("No subsections to plot")
        return
      if not colors:
        colors = ['red','blue','green','orange','purple','brown','pink','gray','olive','cyan'][:len(subsections)]
      x = np.array(range(1, x_center + default_after_end_window + 1))
      xticklabels = [x_i - x_center for x_i in x]
      xticklabels[x_center-1] = ""
      avg_alpha = 0.3
      fit_alpha = 0.7
      
      for feat in list_features:
        print("\nCreating Plot for",feat)
        plt.clf()
        fig, ax1 = plt.subplots(nrows=1, ncols=1) # Horizontally stacked plots
        fig.set_size_inches(8, 6)
        ax1.axvline(x=x_center, color='black', linestyle='--')
        for i, (subsection_name, subsection_counties) in enumerate(subsections.items()):
          subsection_color = colors[i]
          x_before_subsection = []
          y_before_subsection = []
          x_after_subsection = []
          y_after_subsection = []
          x_subsection = []
          y_subsection = []
          for target in subsection_counties:
            if target not in y_before_county[feat] or target not in y_after_county[feat]:
              continue
            x_before_subsection.append(x_before_county[feat][target])
            y_before_subsection.append(y_before_county[feat][target])
            x_after_subsection.append(x_after_county[feat][target])
            y_after_subsection.append(y_after_county[feat][target])
            x_subsection.append(x_county[feat][target])
            y_subsection.append(y_county[feat][target])
          print("-> Subsection",subsection_name)
          x_before_spaghetti_feat = np.array(x_before_subsection)
          y_before_spaghetti_feat = np.array(y_before_subsection)
          x_after_spaghetti_feat = np.array(x_after_subsection)
          y_after_spaghetti_feat = np.array(y_after_subsection)
          #x_spaghetti_feat = np.array(x_subsection)
          y_spaghetti_feat = np.array(y_subsection)
          if event_buffer > 0:
            x_before_spaghetti_feat = x_before_spaghetti_feat[:,:-event_buffer]
            y_before_spaghetti_feat = y_before_spaghetti_feat[:,:-event_buffer]
            x_after_spaghetti_feat = x_after_spaghetti_feat[:,event_buffer:]
            y_after_spaghetti_feat = y_after_spaghetti_feat[:,event_buffer:]
          x_before = x_before_spaghetti_feat[0]
          x_after = x_after_spaghetti_feat[0]
          # Plot average of All
          ax1.plot(x, np.nanmean(y_spaghetti_feat, axis=0), color=subsection_color, linestyle='--', alpha=avg_alpha, label='{} Average'.format(subsection_name))
          # Linear Regression Line Before, After, All
          try:
            regression_before = LinearRegression().fit( x_before.reshape(-1, 1), np.nanmean(y_before_spaghetti_feat, axis=0) )
            ax1.plot(x_before, regression_before.predict(x_before.reshape(-1, 1)), color=subsection_color, alpha=fit_alpha)
            regression_after = LinearRegression().fit( x_after.reshape(-1, 1), np.nanmean(y_after_spaghetti_feat, axis=0) )
            ax1.plot(x_after, regression_after.predict(x_after.reshape(-1, 1)), color=subsection_color, alpha=fit_alpha)
          except Exception as e:
            print("Error with plotting",feat)
            print(e)
            continue
        # Set x label for ax1
        ax1.set_xlabel("Weeks Before and After {}".format(event_name))
        fig.suptitle('Effect of {} Across {}'.format(event_name,subsection_str))
        y_ses_all = y_ses[feat][1]
        y_ses_all.extend(y_ses[feat][2])
        y_ses_all.extend(y_ses[feat][3])
        ymin = np.nanmin(y_ses_all) - 0.05
        ymax = np.nanmax(y_ses_all) + 0.05
        # Format plot
        plt.ylabel("{} Z-Score".format(feat))
        for ax in [ax1]:
          ax.set_xticks(x)
          ax.set_xticklabels(xticklabels, rotation=0)
          ax.axis(ymin=ymin, ymax=ymax)
          ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt_name = "rdd_plots_all/rdd_{}_{}_{}_{}.png".format(
          num_counties_studied, subsection_str, feat.lower(), event_name.lower().replace(" ","_")
        )
        plt.savefig(filename=plt_name)  
        
    # Create a plot for each subsection
    ses_1_counties = [target for target in county_to_ses.keys() if county_to_ses[target] == 1]
    ses_2_counties = [target for target in county_to_ses.keys() if county_to_ses[target] == 2]
    ses_3_counties = [target for target in county_to_ses.keys() if county_to_ses[target] == 3]
    plot_by_subsection({"SES 1":ses_1_counties,"SES 2":ses_2_counties,"SES 3":ses_3_counties}, "SES", colors=['red','blue','green'])
        
    
    # Create a pandas dataframe
    print("\nCreating Pandas Dataframe of Outcomes")
    columns = ['fips','county','feat','event_name','event_date','event_date_centered','target_before','target_after','target_before_slope','target_after_slope','target_before_intercept','target_after_intercept','target_diff_slope','target_diff_intercept','slope_effect','intercept_effect']
    output = pd.DataFrame(output_dicts)[columns]
    output = output.dropna()
    print(output.sort_values(by=['fips','feat']))
    
    # Merge the SES data
    output = output.merge(ses, how='left', on='fips')
    output_ses_1 = output[output['ses3'] == 1]
    output_ses_2 = output[output['ses3'] == 2]
    output_ses_3 = output[output['ses3'] == 3]
    
    def print_outputs(output_df):
      output_df = output_df.copy(deep=True)
      outcomes  = ['slope_effect','intercept_effect']
      for feat in list_features:
        print("\nMean findings for",feat)
        outcomes_feat = output_df[output_df['feat'] == feat]
        for outcome in outcomes:
          dat = outcomes_feat[outcome].replace([np.inf, -np.inf], np.nan).dropna()
          print("-> Mean (stderr) {} = {} ({})".format( outcome, round(np.mean(dat),4), round(np.std(dat)/np.sqrt(len(dat)),4) ))
    
    print("\nResults for all SES levels --------- ", output.shape)
    print_outputs(output)
    print("\nResults for SES Level 1 --------- ", output_ses_1.shape)
    print_outputs(output_ses_1)
    print("\nResults for SES Level 2 --------- ", output_ses_2.shape)
    print_outputs(output_ses_2)
    print("\nResults for SES Level 3 --------- ", output_ses_3.shape)
    print_outputs(output_ses_3)
    
    # Correlation between SES and Intervention Effect
    print("\nCorrelation between SES and outcomes")
    corr_columns = ['pblack','pfem','p65old','phisp','unemployment_rate_2018','ses','svar','lnurban','ses3']
    for feat in list_features:
      print("\nCorrelation for",feat)
      outcomes_feat = output[output['feat'] == feat]
      outcome = 'intercept_effect'
      dat = outcomes_feat[outcome].replace([np.inf, -np.inf], np.nan).dropna()
      for col in corr_columns:
        corr = np.corrcoef(dat,output[col].loc[dat.index])[0,1]
        print("-> Correlation between {} and {} = {}".format(outcome,col,round(corr,4)))
    
    # Write the output to a CSV
    output.to_csv("rdd_{}_data.csv".format(event_name.lower().replace(" ","_")),index=False)