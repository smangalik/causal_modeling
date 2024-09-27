"""
load the virtual environment: `source /data/smangalik/myenvs/diff_in_diff/bin/activate`
run as `python3.5 generate_diff_in_diffs.py` or `python3.5 generate_diff_in_diffs.py topics`
"""

import glob
import os
import sys  # noqa: F401
from typing import List, Tuple
from pymysql import cursors, connect # type: ignore
from collections import defaultdict

import warnings
import argparse
from tqdm import tqdm
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Process feature table for diff in diff analysis")
parser.add_argument('--topics', dest="topics", default=False ,action='store_true', help='Is the analysis done on topics?')
args = parser.parse_args()

# is the analysis being done on topics? (topic_num needs to be interpreted)
topics = args.topics
print("topics mode?",topics)

# Where to load data from
#data_file = "/data/smangalik/featANS.dd_daa_c2adpt_ans_nos.timelines19to20_lex_3upts.yw_cnty.wt50.05fc.csv" # from apollo
data_file = "/data/smangalik/lbmha_yw_cnty.csv" # from research repo

# The county factors we want to cluster counties on
# TODO improve the set of features
'''
Table: county_data.countyHealthRankings2020_v1je1
Table: county_data.[2021 tables]
Documentation of years/sources everything is taken from: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation
'''
# county_factors_table = "county_disease.county_PESH"
# county_factors_fields = "percent_male10, med_age10, log_med_house_income0509, high_school0509, bac0509"
# county_factors_fields += ",log_pop_density10, percent_black10,percent_white10, foreign_born0509, rep_pre_2012, married0509"

county_factors_table = "ctlb2.county_PESH_2020"
county_factors_fields = "perc_republican_president_2016, perc_republican_president_2020, perc_other_president_2020, PercPop25OverBachelorDeg_census19, Log10MedianHouseholdIncome_chr20, UnemploymentRate_bls20, PercHomeowners_chr20, SocialAssociationRate_chr20, ViolentCrimeRate_chr20, PercFairOrPoorHealth_chr20, AgeAdjustedDeathRate_chr20, SuicideRateAgeAdjusted_chr20, PercDisconnectedYouth_chr20"

# Number of principal componenets used to compress sociodemographic data
pca_components = 3

# How many of the top populous counties we want to keep
top_county_count = 500 # 3232 is the maximum number, has a serious effect on results

# Number of nearest neighbors
k_neighbors = 10

# Diff in Diff Windows
default_before_start_window = 1 # additional weeks to consider before event startf
default_after_end_window = 1 # additional weeks to consider after event end
default_event_buffer = 1 # number of weeks to ignore before and after event

# Confidence Interval Multiplier
ci_window = 1.96

# Scale factor for y-axis
scale_factor = 100000

# event_date_dict[county] = [event_start (datetime, exclusive), event_end (datetime, inclusive), event_name]
county_events = {}
county_events['11000'] = [datetime.datetime(2020, 5, 25), datetime.datetime(2020, 6, 21), "Death of George Floyd"]
county_events['40143'] = [datetime.datetime(2020, 5, 25), None, "Death of George Floyd"]

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

# Populate county adjacency list with all adjacent counties
print("Loading county adjacency list")
county_adjacency = {}
with open("/users2/smangalik/causal_modeling/county_adjacency_2024.txt") as countyAdjacency:
    lines = countyAdjacency.read().splitlines()[1:] # read and skip header
    for line in tqdm(lines):
      county_name, geo_id, neighbor_name, neighbor_geo_id = line.split("|")
      if geo_id not in county_adjacency.keys():
        county_adjacency[geo_id] = {geo_id} # add self to adjacency list
      county_adjacency[geo_id] = county_adjacency[geo_id].union({neighbor_geo_id})
print("Loaded adjacency list with",len(county_adjacency),"counties")
print("Example adjacency list for 01001",county_adjacency['01001'])

# Populate lat/long for each county
print("Loading county lat/long")
county_lat_long = {}
with open("/users2/smangalik/causal_modeling/uscounties_lat_long.csv") as countyLatLong:
  lines = countyLatLong.read().splitlines()[1:] # read and skip header
  for line in lines:
    county_name,fips,state_iabbr,state_name,lat,long,population = line.split(",")
    fips = str(fips).zfill(5)
    county_lat_long[fips] = np.array([float(lat),float(long)])

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

# Get the static factors for the most populous counties
def get_static_factors(cursor, populous_counties, pca_components=pca_components):
  demo_factors_height = len(populous_counties)
  demo_factors_width = len(county_factors_fields.split(','))
  demo_factors = np.zeros((demo_factors_height, demo_factors_width))

  # Get PESH factors
  for i, cnty in enumerate(populous_counties):
    sql = "select {} from {} where cnty = {};".format(
      county_factors_fields,
      county_factors_table,
      cnty
    )
    cursor.execute(sql)
    result = cursor.fetchone()
    # Write to demo_factors
    demo_factors[i,] = np.asarray(result, dtype=np.float32)
  # replace NaNs with column means
  for i in range(demo_factors.shape[1]):
    col = demo_factors[:,i]
    mean = np.nanmean(col)
    col[np.isnan(col)] = mean  
  # Fit a StandardScaler and transform the demo factors
  scaler = StandardScaler()
  scaler.fit(demo_factors)
  demo_factors = scaler.transform(demo_factors)
  # Take the PCA of the county factors to calculate neighbors
  pca = PCA(n_components=pca_components, random_state=0)
  pca.fit(demo_factors)
  demo_factors = pca.transform(demo_factors)
  
  # Get Lat/Long factors
  location_factors = np.zeros((len(populous_counties),2),dtype=np.float32)
  for i, cnty in enumerate(populous_counties):
    location_factors[i,] = county_lat_long.get(cnty,(0,0))
  
  # Combine all factors
  static_factors = np.concatenate((demo_factors,location_factors),axis=1)
  
  return static_factors
  

# Get county features relative to some target county for all populous counties
def get_county_factors(populous_counties, static_factors, target_county='36103', feat='DEP'):
  target_factors_height = len(populous_counties) 
  target_factors_width = 4
  target_factors = np.zeros((target_factors_height, target_factors_width))
  
  neighbors = NearestNeighbors(n_neighbors=k_neighbors)
  
  # Find the target county's before-event dates
  blank_event = [None,None,None]
  target_event_start, target_event_end, event_name = first_covid_case.get(target_county,blank_event)
  if target_event_start is None:
    county_factors = np.concatenate((static_factors, np.zeros((target_factors_height,1))),axis=1)
    neighbors.fit(county_factors)
    return county_factors, neighbors
  target_before, target_after, dates_before, dates_after = feat_usage_before_and_after(target_county, feat, event_start=target_event_start, event_end=target_event_end)

  for i, cnty in enumerate(populous_counties):
    # Is the county adjacent to the target county?
    adjacent_counties = county_adjacency.get(target_county,set())
    county_adjacent = 1.0 if cnty in adjacent_counties else 0.0
    
    # What is the feature usage for the county before the event?
    feat_score_before_event = avg_feat_usage_from_dates(county=cnty, dates=dates_before, feat=feat)
    
    # print("DEBUGGING COUNTY FACTORS")
    # print("adjacent_counties",adjacent_counties,"county_adjacent",county_adjacent)
    # print("feat_score_before_event",feat_score_before_event,"from",cnty,dates_before,feat)
    
    # Write to target_factors
    target_factors[i,] = np.asarray([county_adjacent,feat_score_before_event,feat_score_before_event,feat_score_before_event], dtype=np.float32)
    
  # replace NaNs with column means
  for i in range(target_factors.shape[1]):
    col = target_factors[:,i]
    mean = np.nanmean(col)
    col[np.isnan(col)] = mean  
    
  # Combine static and target factors
  county_factors = np.concatenate((static_factors,target_factors),axis=1)

  # Fit the nearest neighbors
  neighbors.fit(county_factors)

  return county_factors, neighbors

# Returns the yearweek that the date is within
def date_to_yearweek(d:datetime.datetime) -> str:
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber).zfill(2)

def yearweek_to_dates(yw:str) -> Tuple[datetime.datetime, datetime.datetime]:
  '''
  Returns the first and last day of a week given as "2020_11"
  '''
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  return monday, sunday

def avg_feat_usage_from_dates(county,dates,feat='DEP') -> float:
  feat_usages = []

  for date in dates:
    if county_feats.get(county,None) is not None:
      if county_feats[county].get(date,None) is not None: 
        if county_feats[county][date].get(feat,None) is not None:
          feats_for_date = np.array(county_feats[county][date][feat])
          feat_usages.append(feats_for_date)
        else:
          #print("No feature",feat,"for",county,"on",date)
          pass
      else:
        #print("No date",date,"for",county)
        pass
    else:
      #print("No county",county)
      pass

  if len(feat_usages) == 0:
    # print("No matching dates for", county, "on dates", dates)
    return None

  return np.mean(feat_usages)


def feat_usage_before_and_after(county, feat, event_start, event_end=None,
                                 before_start_window=default_before_start_window,
                                 after_start_window=default_after_end_window,
                                 event_buffer=default_event_buffer) -> Tuple[float,float,List[str],List[str]]:
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
  for i in range(1,before_start_window + 2):
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
  return avg_feat_usage_from_dates(county, before_dates, feat), avg_feat_usage_from_dates(county, after_dates, feat), before_dates, after_dates

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
    df['feat'] = df['score_category'].str.replace('_SCORE','')
    score_col = 'score'

    # Require n_users >= 200
    df = df[df['n_users'] >= 200]
    
    print("Data from CSV:", data_file)
    print(df)
    print("Columns:",df.columns.tolist())

    list_features = df['feat'].unique()
    
    # Normalize wavg_score to be between 0 and 1 per county feature
    df[score_col] = df.groupby(['cnty','feat'])[score_col].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    print("Loading Unique Features:",list_features)
    county_feats = defaultdict(lambda: defaultdict(dict))
    grouped = df.groupby(['feat', 'cnty', 'yearweek'])
    for (feat, county, yearweek), group in tqdm(grouped):
        county_feats[county][yearweek][feat] = group[score_col].iloc[0]
    county_feats = {k: dict(v) for k, v in county_feats.items()} # Convert to a regular dict
    
    print("county_feats['40143']['2020_07']['DEP'] =",county_feats['40143']['2020_07']['DEP'])
    print("county_feats['40143']['2020_08']['DEP'] =",county_feats['40143']['2020_08']['DEP'])
    print("county_feats['40143']['2020_20']['ANX'] =",county_feats['40143']['2020_20']['ANX'])
    print("county_feats['40143']['2020_20']['ANG'] =",county_feats['40143']['2020_20']['ANG'])
    available_yws = list(county_feats['40143'].keys())
    available_yws.sort()
    print("\nAvailable weeks for 40143:",  available_yws)
    
    # Display nearest neighbors for first county
    #test_county = '36103' # Suffolk, NY
    test_county = '40143' # Washington, DC
    test_county_index = list(populous_counties).index(test_county)
    
    # Create county_factor matrix and n-neighbors mdoel
    static_factors = get_static_factors(cursor, populous_counties)
    # TODO dynamically determine the target county
    sample_county_factors, sample_neighbors = get_county_factors(populous_counties, static_factors, target_county=test_county, feat='DEP')
    print("\nFactors for county {} = {}\n".format(test_county,sample_county_factors[test_county_index]))
    dist, n_neighbors = sample_neighbors.kneighbors([sample_county_factors[test_county_index]], 11, return_distance=True)
    for i, n in enumerate(n_neighbors[0]):
        print('The #{}'.format(i),'nearest County is',populous_counties[n],'with distance',dist[0][i])

    # Get the closest k_neighbors for each populous_county we want to examine
    county_representation = {}
    matched_counties = {}

    # Calculate Average and Weighted Average Weekly Feat Score for Baselines
    county_list = county_feats.keys() # all counties considered in order
    county_list_weights = [county_representation.get(c,1) for c in county_list] # weight based on neighbor count
    avg_county_list_usages = {} # avg_county_list_usages[yearweek][feat] = average value
    all_possible_yearweeks = sorted(df['yearweek'].unique().tolist())
    for yearweek in all_possible_yearweeks:
      for feat in list_features:
        feat_usages = []
        for county in county_list:
          if county_feats.get(county,None) is not None:
            if county_feats[county].get(yearweek,None) is not None: 
              if county_feats[county][yearweek].get(feat,None) is not None:
                feats_for_date = np.array(county_feats[county][yearweek][feat])
                feat_usages.append(feats_for_date)
        if len(feat_usages) == 0:
          # print("No matching dates for", county, "on dates", dates)
          continue
        avg_county_list_usages[yearweek] = avg_county_list_usages.get(yearweek,{})
        avg_county_list_usages[yearweek][feat] = np.nanmean(feat_usages)
        
    print("\nAverage Feature Usage for all counties")
    for yearweek in all_possible_yearweeks:
      print(yearweek,avg_county_list_usages[yearweek])

    # Capture the neighbors for each populous county by feat
    print("\nCalculating Neighbors for",len(populous_counties),"counties")
    for target in tqdm(populous_counties):
      matched_counties[target] = {}
      for feat in list_features:
        # county event date for target, skip if no event
        target_county_event, _, _ = first_covid_case.get(target,[None,None,None])
        
        # Get the county factors and neighbors
        county_factors, neighbors = get_county_factors(populous_counties, static_factors, target_county=target, feat=feat)

        # Get the top-k neighbors
        county_index = list(populous_counties).index(target)
        factor_to_compare = county_factors[county_index]
        n_neighbors = neighbors.kneighbors([factor_to_compare], k_neighbors * 2, return_distance=False)
        matched_counties[target][feat] = []

        # pick the k_neighbors closest neighbors
        for i, n in enumerate(n_neighbors[0][1:]): # skip 0th entry (self)
          # Stop when we have enough neighbors
          if len(matched_counties[target][feat]) >= k_neighbors:
            break
          ith_closest_county = populous_counties[n]
          # Don't match with counties that have no relevant data
          ith_closest_county_event, _, _ = first_covid_case.get(ith_closest_county,[None,None,None])
          if ith_closest_county_event is None:
            continue
          # filter out counties with county events close by in time
          if date_to_yearweek(target_county_event) == date_to_yearweek(ith_closest_county_event):
            continue
          # determine how much each county appears
          if ith_closest_county not in county_representation.keys():
            county_representation[ith_closest_county] = 0
          county_representation[ith_closest_county] += 1
          matched_counties[target][feat].append(ith_closest_county)

    neighbor_counts = sorted(county_representation.items(), key=lambda kv: kv[1])
    print("\nCount of times each county is a neighbor\n", neighbor_counts[:10],"...",neighbor_counts[-10:], '\n')


    # Calculate diff in diffs dict[county] = value
    target_befores = {}
    target_afters = {}
    target_diffs = {}
    matched_befores = {}
    matched_afters = {}
    matched_diffs = {}
    avg_matched_befores = {}
    std_matched_befores = {}
    avg_matched_afters = {}
    std_matched_afters = {}
    avg_matched_diffs = {}
    std_matched_diffs = {}
    
    percent_change_map = {}
    stderr_change_map = {}
    amount_change_map = {}

    # Delete all previous plots
    print("Deleting all previous plots...")
    files = glob.glob('covid_plots/*')
    for f in files:
      os.remove(f)
    
    print("\nCalculating Diff in Diffs for",len(populous_counties),"counties")
    for target in tqdm(populous_counties):

      # George Flyod's Death
      #target_event_start, target_event_end, event_name = county_events.get(target,[None,None,None])
      # First Covid Death
      #target_event_start, target_event_end, event_name = first_covid_death.get(target,[None,None,None])
      
      # First Case of Covid
      blank_event = [None,None,None]
      target_event_start, target_event_end, event_name = first_covid_case.get(target,blank_event)

      # If no event was found for this county, skip
      if target_event_start is None and target_event_end is None:
        print("Skipping {} ({}) due to missing event".format(target,fips_to_name.get(target)))
        continue # no event was found for this county
      
      #print()
      for feat in list_features: # run diff-in-diff against each features
        
        target_before, target_after, dates_before, dates_after = feat_usage_before_and_after(target, feat, event_start=target_event_start, event_end=target_event_end)
        if target_before is None or target_after is None:
          #print("Skipping {} ({}) for {} due to missing data".format(target,fips_to_name.get(target),feat))
          continue

        # How the target county changed
        target_diff = np.subtract(target_after,target_before)
        target_befores[target] = target_before
        target_afters[target] = target_after
        target_diffs[target] = target_diff
        
        # How the matched counties changed
        matched_diffs[target] = []
        matched_befores[target] = []
        matched_afters[target] = []
        for matched_county in matched_counties[target][feat]:
          matched_before, matched_after, _, _ = feat_usage_before_and_after(matched_county, feat, event_start=target_event_start, event_end=target_event_end)
          if matched_before is None or matched_after is None: 
            continue
          matched_diff = np.subtract(matched_after,matched_before)
          # Add all differences, then divide by num of considered counties
          matched_diffs[target].append(matched_diff)
          matched_befores[target].append(matched_before)
          matched_afters[target].append(matched_after)
        if matched_befores[target] == [] or matched_afters[target] == [] or matched_diffs[target] == []:
          print("Skipping {} ({}) for {} due to missing match data".format(target,fips_to_name.get(target),feat))
          continue # this target-feat is not viable (no match data)
        # Average/std change from all matched counties
        avg_matched_befores[target] = np.mean(matched_befores[target])
        std_matched_befores[target] = np.std(matched_befores[target])
        avg_matched_afters[target] = np.mean(matched_afters[target])
        std_matched_afters[target] = np.std(matched_afters[target])
        avg_matched_diffs[target] = np.mean(matched_diffs[target])
        std_matched_diffs[target] = np.std(matched_diffs[target])

        # Diff in Diff Calculation
        target_expected = target_before + avg_matched_diffs[target]
        intervention_effect = target_after - target_expected
        intervention_percent = round(intervention_effect/target_expected * 100.0, 2)

        # print("County: {} ({}) studying {} against {}".format(target, fips_to_name.get(target), feat, event_name))
        
        matched_neighbor_count = len(matched_befores[target])
        stderr_match_before = std_matched_befores[target] / np.sqrt(matched_neighbor_count)
        stderr_match_after = std_matched_afters[target] / np.sqrt(matched_neighbor_count)
        stderr_match_diff = std_matched_diffs[target] / np.sqrt(matched_neighbor_count)
        
        increase_decrease = "increased" if intervention_effect > 0 else "decreased"
        
        stderr_change = intervention_effect/stderr_match_after
        # print("-> Change in {} {} by {}% ({} stderrs)".format(feat, increase_decrease, intervention_percent, stderr_change))
        
        # Relevant Dates
        begin_before, _ = yearweek_to_dates(min(dates_before))
        _, end_before = yearweek_to_dates(max(dates_before))
        begin_after, _ = yearweek_to_dates(min(dates_after))
        _, end_after = yearweek_to_dates(max(dates_after))
        middle_before = begin_before + (end_before - begin_before)/2
        middle_after = begin_after + (end_after - begin_after)/2
        middle_middle = middle_before + (middle_after - middle_before)/2
        
        # Write findings to maps
        map_key = "{}:{}:{}".format(target,feat,middle_middle)
        percent_change_map[intervention_percent] = map_key
        stderr_change_map[map_key] = stderr_change
        amount_change_map[intervention_effect] = [target,feat,middle_middle]

        is_significant = abs(intervention_effect) > stderr_match_diff*ci_window
        if not is_significant:
          # print("NOT significant: County {} ({}) for {}, {} was not greater than {}\n --- ".format(
          #   feat,target,event_name,abs(intervention_effect),stderr_match_diff*ci_window)
          # )
          continue # only plot significant results
        else:
          # print("Target Before:                                 ", target_before)
          # print("Target After (with intervention / observation):", target_after)
          # print("Target After (without intervention / expected):", target_expected)
          # print("Intervention Effect:                           ", intervention_effect)
          # print("From {} matches, giving {} matched befores and {} matched afters".format(
          #   len(matched_counties[target][feat]), len(matched_befores[target]),len(matched_afters[target])))
          # print("Plotting: County {} ({}) for {}\n --- ".format(feat,target,event_name))  
          pass     

        # --- Plotting ---

        # Calculate in-between dates and xticks
        x = [middle_before, middle_after]
        xticks = [begin_before, end_before, begin_after, end_after]

        # Confidence Intervals
        ci_down = [target_before-stderr_match_before, target_expected-stderr_match_after] # One stderr up
        ci_up = [target_before+stderr_match_before, target_expected+stderr_match_after] # One stderr down
        ci_down_2 = [target_before-stderr_match_before*ci_window, target_expected-stderr_match_after*ci_window] # Two stderrs up
        ci_up_2 = [target_before+stderr_match_before*ci_window, target_expected+stderr_match_after*ci_window] # Two stderrs down

        # Create DiD Plot
        plt.clf() # reset old plot
        x = [2, 6]
        xticks = [1, 3, 4, 5, 7]
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # Horizontally stacked plots
        fig.set_size_inches(14, 6)
        ax1.plot(x, [target_before, target_after], 'b-', label='Target (Observed)')
        ax1.plot(x, [target_before, target_expected],'c--', label='Target (Expected)')
        ax1.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
        ax1.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
        ax1.plot([x[1],x[1]], [target_after, target_expected], 'k-', \
          label='Intervention Effect ({})'.format(round(intervention_effect,3)))
        
        matched_befores_target = matched_befores[target][:k_neighbors]
        matched_afters_target = matched_afters[target][:k_neighbors]
        ax2.plot([x[0]]*len(matched_befores_target), matched_befores_target, 'r+', alpha=0.5)
        ax2.plot([x[1]]*len(matched_afters_target), matched_afters_target, 'r+', alpha=0.5)
        ax2.plot(x,[avg_matched_befores[target],avg_matched_afters[target]],'r--',label="Average Match", alpha=0.8)
        
        #plt.axhline(y=avg_county_list_usages[feat], color='g', linestyle='-',label='Average {}'.format(feat))
        #plt.axhline(y=weighted_avg_county_list_usages[feat], color='g', linestyle='--',label='Weighted Average {}'.format(feat))
        fig.suptitle("{}'s {} Before/After {}".format(fips_to_name[target], feat, event_name))
        
        # Plot the average and weighted average per week
        x = [2,6]
        y_vals = [
          np.mean([ avg_county_list_usages[yw][feat] for yw in dates_before ]),
          np.mean([ avg_county_list_usages[yw][feat] for yw in dates_after]),
        ]
        ax2.plot(x, y_vals, 'g--', label='Average Overall', alpha=0.8)

        # Plot the target county per week
        x_pos = np.arange(1,8)
        y_pos = [begin_before, end_before, middle_before, middle_middle, middle_after, begin_after, end_after]
        x_pos = np.arange(1,8)
        y_vals = [county_feats.get(target,{}).get(date_to_yearweek(yw),{}).get(feat,np.nan) for yw in y_pos]
        #ax2.plot(x_pos, y_vals, 'b-', label='Target {}'.format(feat), alpha=0.3)

        # Plot the matched counties per week
        x_pos = np.arange(1,8)
        y_vals = np.zeros((len(matched_counties[target][feat]),len(y_pos)))
        for i, matched_county in enumerate(matched_counties[target][feat]):
          y_vals[i] = [county_feats.get(matched_county,{}).get(date_to_yearweek(yw),{}).get(feat,np.nan) for yw in y_pos]
        y_vals = np.nanmean(y_vals,axis=0)
        #ax2.plot(x_pos, y_vals, 'r-', label='Matched {}'.format(feat), alpha=0.3)

        # Format plot
        plt.ylabel("Change in {}".format(feat))
        for ax in [ax1, ax2]:
          ax.set_xticks(xticks)
          ax.set_xticklabels([
            "{} Weeks Before".format(default_event_buffer + default_before_start_window + 1),
            "{} Week Before".format(default_event_buffer),
            middle_before.strftime("%b %-d, %Y"),
            "{} Week After".format(default_event_buffer),
            "{} Weeks After".format(default_event_buffer + default_after_end_window + 1)
          ], rotation=45)
          ax.axis(ymin=0, ymax=1)
          ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt_name = "covid_plots/did_{}_{}_before_after_covid_case.png".format(target, feat)

        plt.savefig(plt_name) # optionally save all figures

    # Print out the results in sorted order
    print("\nSorted Results for COVID Case Changes:")
    for percent_change in sorted(percent_change_map.keys()):
      feature = percent_change_map[percent_change]
      stderr = stderr_change_map[feature]
      print("{} actual diff was {} stderr's larger than the counterfactual diff".format(feature, round(stderr,2)))
    print("\nSummary for COVID Case Change:")
    for feat in list_features:
      feat_diffs = [key for key,value in percent_change_map.items() if value.split(":")[1] == feat]
      print("-> {} average diff was {} stderr's larger than the counterfactual diff".format(feat, np.mean(feat_diffs)))
    all_diffs = list(percent_change_map.keys())
    print("-> Overall average diff was {} stderr's larger than the counterfactual diff".format(np.mean(all_diffs)))
    
    # Plot amount change over time
    for feat in list_features:
      plt.clf()
      fig, ax = plt.subplots()
      fig.set_size_inches(6, 6)
      for amount_change, key in sorted(amount_change_map.items(), key=lambda x: x[1][2]): # sort by middle_middle
        # Plot the amount_change over middle_middle
        _, feat_here, middle_middle = key
        if feat_here != feat:
          continue
        plt.plot(middle_middle, amount_change, 'b+', alpha=0.3)
      plt.title("{} Amount Change Over Time".format(feat))
      plt.ylabel("{} Change".format(feat))
      plt.xlabel("Event Date")
      # rotate x-axis labels
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()
      plt_name = "over_time_changes/amount_change_{}_before_after_covid_case.png".format(feat)
      plt.savefig(plt_name)
      print("Saved",plt_name)