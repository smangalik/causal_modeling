"""
load the virtual environment: `source /data/smangalik/myenvs/diff_in_diff/bin/activate`
run as `python3.5 generate_diff_in_diffs.py` or `python3.5 generate_diff_in_diffs.py topics`
"""

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
from scipy import stats


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
data_file = "/data/smangalik/featANS.dd_daa_c2adpt_ans_nos.timelines19to20_lex_3upts.yw_cnty.wt50.05fc.csv"

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

# Number of principal componenets used in PCA
pca_components = 10

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

# Populate county adjacency list into a binary numpy array
print("Loading county adjacency list")
county_adjacency = {}
with open("/users2/smangalik/causal_modeling/county_adjacency_2024.txt") as countyAdjacency:
    lines = countyAdjacency.read().splitlines()[1:] # read and skip header
    possible_geo_ids = set()
    for line in lines:
      county_name, geo_id, neighbor_name, neighbor_geo_id = line.split("|")
      possible_geo_ids.add(geo_id)
    possible_geo_ids = sorted(list(possible_geo_ids))
    for line in tqdm(lines):
      county_name, geo_id, neighbor_name, neighbor_geo_id = line.split("|")
      if geo_id not in county_adjacency.keys():
        county_adjacency[geo_id] = np.zeros(len(possible_geo_ids),dtype=np.float32)
      neighbor_index = possible_geo_ids.index(neighbor_geo_id)
      county_adjacency[geo_id][neighbor_index] = 1.0   
print("Loaded adjacency list with",len(county_adjacency),"counties and",len(possible_geo_ids),"neighbors")
print("Example adjacency list for 01001",county_adjacency['01001'])

print('Connecting to MySQL...')

# Open default connection
connection  = connect(read_default_file="~/.my.cnf")

# Get the 100 most populous counties
def get_populous_counties(cursor, base_year) -> list:
  populous_counties = []
  sql = "select * from ctlb.counties{} order by num_of_users desc limit {};".format(base_year,top_county_count)
  cursor.execute(sql)
  for result in cursor.fetchall():
    cnty, num_users = result
    cnty = str(cnty).zfill(5)
    populous_counties.append(cnty)
  return populous_counties

# Get county features like [age, income, education, (race, ethnicity, well_being)]
def get_county_factors(cursor, base_year, relevant_counties, normalize=False):
  county_factors_height = len(relevant_counties) 
  county_factors_width = len(county_factors_fields.split(',')) + len(possible_geo_ids)
  county_factors = np.zeros((county_factors_height, county_factors_width))

  for i, cnty in enumerate(relevant_counties):
      # # Get stats
      # # TODO Use CTLB to get better stats on user tweets [cnty|num_users|num_tweets]
      # sql = "select * from county_topic_change.msgs_100u_{} where cnty = {};".format(base_year,cnty)
      # cursor.execute(sql)
      # result = cursor.fetchone()
      # if result is None:
      #     continue
      # cnty, num_users, num_tweets = result
      # county_factors[i][0] = np.log(float(num_users)) # log number of users
      # county_factors[i][1] = np.log(float(num_tweets)) # log number of tweets

      # Get PESH factors
      sql = "select {} from {} where cnty = {};".format(
        county_factors_fields,
        county_factors_table,
        cnty
      )
      cursor.execute(sql)
      result = cursor.fetchone()
      demo_factors = np.asarray(result,dtype=np.float32)
      
      # Add adjacency list
      county_adjacencies = county_adjacency.get(cnty,np.zeros(len(possible_geo_ids),dtype=np.float32))
      
      # Combine all factors
      county_factors[i] = np.concatenate((demo_factors,county_adjacencies))

  # replace NaNs with column means
  for i in range(county_factors.shape[1]):
    col = county_factors[:,i]
    mean = np.nanmean(col)
    col[np.isnan(col)] = mean
  
  relevant_counties = np.asarray(relevant_counties)

  # Normalize columns of county_factors via z-score
  if normalize:
      county_factors = stats.zscore(county_factors, axis=0)
      
  # Fit a StandardScaler and transform the county factors
  scaler = StandardScaler()
  scaler.fit(county_factors)
  county_factors = scaler.transform(county_factors)

  # Take the PCA of the county factors to calculate neighbors
  pca = PCA(n_components=pca_components, random_state=0)
  pca.fit(county_factors)
  county_factors = pca.transform(county_factors)

  # Fit the nearest neighbors
  neighbors = NearestNeighbors(n_neighbors=k_neighbors)
  neighbors.fit(county_factors)

  return county_factors, neighbors, relevant_counties

# Get category map from topic number to top words
def get_topic_map(cursor) -> dict:
  topic_map = {}
  sql = 'select category, group_concat(term order by weight desc) as "terms" from dlatk_lexica.met_a30_2000_freq_t50ll group by category;'
  cursor.execute(sql)
  for result in cursor.fetchall():
    category, terms = result
    topic_map[category] = terms.split(',')[:10]
  return topic_map

# Iterate over all time units to create county_feats[county][year_week] = feats
def get_county_feats(cursor, table_years):
  county_feats = {}
  for table_year in table_years:
    print('Processing {}'.format(table_year))

    # sql = "select * from ctlb2.feat$cat_met_a30_2000_cp_w$timelines{}$yw_cnty$1gra;".format(table_year)
    sql = "select * from ctlb2.feat$dd_depAnxLex$timelines{}$yw_cnty$1gra;".format(table_year)
    cursor.execute(sql)

    for result in tqdm(cursor.fetchall_unbuffered()): # Read _unbuffered() to save memory
      yw_county, feat, value, value_norm = result

      if feat == '_int': 
        continue
      feat = int(feat)
      yearweek, county = yw_county.split(":")
      if county == "" or yearweek == "": 
        continue
      county = str(county).zfill(5)

      # Store county_feats
      if county_feats.get(county) is None:
        county_feats[county] = {}
      if county_feats[county].get(yearweek) is None:
        county_feats[county][yearweek] = {}
      try:
        county_feats[county][yearweek][feat] = value_norm
      except IndexError:
        print("IndexError for",county,yearweek,feat)

  return county_feats

# Returns the yearweek that the date is within
def date_to_yearweek(d:datetime.datetime) -> str:
  year, weeknumber, weekday = d.date().isocalendar()
  return str(year) + "_" + str(weeknumber)

# Returns the first and last day of a week given as "2020_11"
def yearweek_to_dates(yw:str) -> Tuple[datetime.datetime, datetime.datetime]:
  year, week = yw.split("_")
  year, week = int(year), int(week)

  first = datetime.datetime(year, 1, 1)
  base = 1 if first.isocalendar()[1] == 1 else 8
  monday = first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))
  sunday = monday + datetime.timedelta(days=6)
  return monday, sunday

# Take in a county and some yearweeks, then average their feat usage
def avg_topic_usage_from_dates(county,dates) -> float:
  feat_usages = []

  for date in dates:
    if county_feats.get(county) is not None and county_feats.get(county).get(date) is not None:
      feats_for_date = np.array(county_feats[county][date]) * scale_factor # scale values and store
      feat_usages.append(feats_for_date)

  if len(feat_usages) == 0:
    # print("No matching dates for", county, "on dates", dates)
    return None

  #print("feats_for_date",feats_for_date)
  #print("feat_usages",feat_usages)

  return np.mean(feat_usages)

def avg_feat_usage_from_dates(county,dates,feat='DEP_SCORE') -> float:
  feat_usages = []

  for date in dates:
    if county_feats.get(county) is not None and county_feats.get(county).get(date) is not None:
      feats_for_date = np.array(county_feats[county][date][feat])
      feat_usages.append(feats_for_date)

  if len(feat_usages) == 0:
    # print("No matching dates for", county, "on dates", dates)
    return None

  return np.mean(feat_usages)

def feat_usage_before_and_after(county, feat, event_start, event_end=None,
                                 before_start_window=default_before_start_window,
                                 after_start_window=default_after_end_window,
                                 event_buffer=default_event_buffer) -> Tuple[float,float,List[str],List[str]]:

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

def plot_diff_in_diff_per_county():

  for feat in list_features:
    matches_before = np.array(matched_befores[target])
    matches_after = np.array(matched_befores[target]) + np.array(matched_diffs[target])
    avg_match_before = np.mean(matches_before)
    avg_match_after = np.mean(matches_after)
    std_match_before = np.std(matches_before)
    std_match_after = np.std(matches_after)
    stderr_match_before = std_match_before / np.sqrt(k_neighbors)
    stderr_match_after = std_match_after / np.sqrt(k_neighbors)

    is_significant = abs(intervention_effect) > stderr_match_after*ci_window
    if not is_significant:
      continue # only plot significant results

    # List significant changes
    increase_decrease = "increased" if intervention_effect > 0 else "decreased"
    print("-> ", feat, increase_decrease, "significantly")

    plt.clf() # reset plot

    # Confidence Intervals
    ci_down = [target_before-stderr_match_before, target_expected-stderr_match_after]
    ci_up = [target_before+stderr_match_before, target_expected+stderr_match_after]
    ci_down_2 = [target_before-stderr_match_before*ci_window, target_expected-stderr_match_after*ci_window]
    ci_up_2 = [target_before+stderr_match_before*ci_window, target_expected+stderr_match_after*ci_window]

    # Create Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.plot(x, [target_before, target_after], 'b-', label='Target (Observed)')
    plt.plot(x, [target_before, target_expected],'c--', label='Target (Expected)')
    #plt.plot([x[0]]*k_neighbors, matches_before, 'r+', alpha=0.2)
    #plt.plot([x[1]]*k_neighbors, matches_after, 'r+', alpha=0.2)
    plt.plot(x,[avg_match_before,avg_match_after],'r--',label='Average Match')
    plt.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
    plt.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
    plt.plot([x[1],x[1]], [target_after, target_expected], 'k--', \
      label='Intervention Effect ({}%)'.format(intervention_percent))
    plt.title("County " + str(target) + " before/after " + event_name)

    # Format plot
    plt.gcf().autofmt_xdate()
    ax.set_xticks(xticks)
    ax.set_xticklabels([
      "{} weeks before event".format(default_event_buffer + default_before_start_window + 1),
      "{} week before event".format(default_event_buffer),
      "{} week after event".format(default_event_buffer),
      "{} weeks after event".format(default_event_buffer + default_after_end_window + 1)
    ])
    plt.xlabel("Time")
    plt.ylabel(str(feat) + " Usage")
    plt.legend()
    plt.tight_layout()

    plt_name = "did_feat_{}_cnty{}_time{}{}{}-{}{}{}.png".format( \
      feat,target,x[0].year,x[0].month,x[0].day,x[1].year,x[0].month,x[0].day)

    plt.savefig(plt_name)

# Read in data with cursor
with connection:
  with connection.cursor(cursors.SSCursor) as cursor:
    print('Connected to',connection.host)

    # Determine the relevant counties
    base_year = 2019
    #populous_counties = get_populous_counties(cursor, base_year)
    #print("\nCounties with the most users in {}".format(base_year),populous_counties[:25],"...")
    populous_counties = sorted(fips_to_population, key=fips_to_population.get, reverse=True)[:top_county_count]
    print("\nCounties with the most users in 2021",populous_counties[:25],"...")


    # Create county_factor matrix and n-neighbors mdoel
    county_factors, neighbors, populous_counties = get_county_factors(cursor, 2016, populous_counties)
    print('\nCounty factor matrix shape:',county_factors.shape)

    # Display nearest neighbors for first county
    test_county = '36103' # Suffolk, NY
    test_county = '40143' # Washington, DC
    test_county_index = list(populous_counties).index(test_county)
    print('\nTest county (',test_county,')\n',county_factors[test_county_index])
    dist, n_neighbors = neighbors.kneighbors([county_factors[test_county_index]], 6, return_distance=True)
    for i, n in enumerate(n_neighbors[0]):
        print('The #{}'.format(i+1),'nearest County is',populous_counties[n],'with distance',dist[0][i])

    # Map topics to their key words
    if topics:
      topic_map = get_topic_map(cursor)
      print()
      print('Topic 0    =',topic_map['0'])
      print('Topic 344  =',topic_map['344'])
      print('Topic 160  =',topic_map['160'])
      print('Topic 1999 =',topic_map['1999'],'\n')


    # Load data from CSV
    df = pd.read_csv(data_file)
    df['cnty'] = df['cnty'].astype(str).str.zfill(5)
    
    print("Data from CSV:", data_file)
    print(df)
    print("Columns:",df.columns.tolist())

    list_features = df['feat'].unique()
    
    print("Loading Unique Features:",list_features)
    county_feats = defaultdict(lambda: defaultdict(dict))
    grouped = df.groupby(['feat', 'cnty', 'yearweek'])
    for (feat, county, yearweek), group in tqdm(grouped):
        county_feats[county][yearweek][feat] = group['wavg_score'].iloc[0]
    county_feats = {k: dict(v) for k, v in county_feats.items()} # Convert to a regular dict
    
    print("county_feats['40143']['2020_19']['DEP_SCORE'] =",county_feats['40143']['2020_19']['DEP_SCORE'])
    print("county_feats['40143']['2020_20']['ANX_SCORE'] =",county_feats['40143']['2020_20']['ANX_SCORE'])
    print("county_feats['40143']['2020_20']['ANG_SCORE'] =",county_feats['40143']['2020_20']['ANG_SCORE'])
    available_yws = list(county_feats['40143'].keys())
    available_yws.sort()
    print("\nAvailable weeks for 40143:",  available_yws)

    # Get the closest k_neighbors for each populous_county we want to examine
    county_representation = {}
    matched_counties = {}

    # Calculate Average and Weighted Average Topic Usage for Baselines
    county_list = county_feats.keys() # all counties considered in order
    county_list_weights = [county_representation.get(c,1) for c in county_list] # weight based on neighbor count
    avg_county_list_usages = {}
    weighted_avg_county_list_usages = {}
    for feat in list_features:
      county_list_usages = [] # all averaged counties stacked
      for county in county_list:
        yearweeks = list(county_feats[county].keys())
        county_list_usages.append( avg_feat_usage_from_dates(county,yearweeks,feat) )
      avg_county_list_usages[feat] = np.average(county_list_usages)
      weighted_avg_county_list_usages[feat] = np.average(county_list_usages, weights=county_list_weights)

    for target in populous_counties:
      
      # county event date for target, skip if no event
      target_county_event, _, _ = first_covid_case.get(target,[None,None,None])

      # Get the top-k neighbors
      county_index = list(populous_counties).index(target)
      n_neighbors = neighbors.kneighbors([county_factors[county_index]], k_neighbors + 1, return_distance=False)
      matched_counties[target] = []

      # pick the #1 closest neighbor with a greedy search
      for i, n in enumerate(n_neighbors[0][1:]): # skip 0th entry (self)

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

        matched_counties[target].append(ith_closest_county)

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
    
    print("\nCalculating Diff in Diffs for",len(populous_counties),"counties")
    for target in populous_counties:

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
      
      print()
      for feat in list_features: # run diff-in-diff against each features
        
        target_before, target_after, dates_before, dates_after = feat_usage_before_and_after(target, feat, event_start=target_event_start, event_end=target_event_end)
        if target_before is None or target_after is None:
          print("Skipping {} ({}) for {} due to missing data".format(target,fips_to_name.get(target),feat))
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
        for matched_county in matched_counties[target]:
          matched_before, matched_after, _, _ = feat_usage_before_and_after(matched_county, feat, event_start=target_event_start, event_end=target_event_end)
          if matched_before is None or matched_after is None: 
            continue
          matched_diff = np.subtract(matched_after,matched_before)
          # Add all differences, then divide by num of considered counties
          matched_diffs[target].append(matched_diff)
          matched_befores[target].append(matched_before)
          matched_afters[target].append(matched_after)
        if matched_befores[target] == []:
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

        print("County: {} ({}) studying {} against {}".format(target, fips_to_name.get(target), feat, event_name))
        
        matched_neighbor_count = len(matched_befores[target])
        stderr_match_before = std_matched_befores[target] / np.sqrt(matched_neighbor_count)
        stderr_match_after = std_matched_afters[target] / np.sqrt(matched_neighbor_count)
        stderr_match_diff = std_matched_diffs[target] / np.sqrt(matched_neighbor_count)

        is_significant = abs(intervention_effect) > stderr_match_diff*ci_window
        if not is_significant:
          print("NOT significant: County {} ({}) for {}, {} was not greater than {}".format(
            feat,target,event_name,abs(intervention_effect),stderr_match_diff*ci_window)
          )
          print(" --- ")
          continue # only plot significant results
        else:
          print("Target Before:                                 ", target_before)
          print("Target After (with intervention / observation):", target_after)
          print("Target After (without intervention / expected):", target_expected)
          print("Intervention Effect:                           ", intervention_effect)
          print("From {} matches, giving {} matched befores and {} matched afters".format(
            len(matched_counties[target]), len(matched_befores[target]),len(matched_afters[target])))
          print("Plotting: County {} ({}) for {}".format(feat,target,event_name))
        increase_decrease = "increased" if intervention_effect > 0 else "decreased"
        stderr_change = intervention_effect/stderr_match_after
        print("-> Change in {} {} by {}% ({} stderrs)".format(feat, increase_decrease, intervention_percent, stderr_change))

        percent_change_map[intervention_percent] = "{}:{}".format(target,feat)
        stderr_change_map["{}:{}".format(target,feat)] = stderr_change

        # Relevant Dates
        begin_before, _ = yearweek_to_dates(min(dates_before))
        _, end_before = yearweek_to_dates(max(dates_before))
        begin_after, _ = yearweek_to_dates(min(dates_after))
        _, end_after = yearweek_to_dates(max(dates_after))
        middle_before = begin_before + (end_before - begin_before)/2
        middle_after = begin_after + (end_after - begin_after)/2

        # Calculate in-between dates and xticks
        x = [middle_before, middle_after]
        xticks = [begin_before, end_before, begin_after, end_after]

        # Confidence Intervals
        ci_down = [target_before-stderr_match_before, target_expected-stderr_match_after]
        ci_up = [target_before+stderr_match_before, target_expected+stderr_match_after]
        ci_down_2 = [target_before-stderr_match_before*ci_window, target_expected-stderr_match_after*ci_window]
        ci_up_2 = [target_before+stderr_match_before*ci_window, target_expected+stderr_match_after*ci_window]

        # Create Plot
        plt.clf() # reset plot
        x = [2, 6]
        xticks = [1, 3, 5, 7]
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        plt.plot(x, [target_before, target_after], 'b-', label='Target (Observed)')
        plt.plot(x, [target_before, target_expected],'c--', label='Target (Expected)')
        matched_befores_target = matched_befores[target][:k_neighbors]
        matched_afters_target = matched_afters[target][:k_neighbors]
        plt.plot([x[0]]*len(matched_befores_target), matched_befores_target, 'r+', alpha=0.2)
        plt.plot([x[1]]*len(matched_afters_target), matched_afters_target, 'r+', alpha=0.2)
        plt.plot(x,[avg_matched_befores[target],avg_matched_afters[target]],'r--',label='Average Match')
        plt.fill_between(x, ci_down, ci_up, color='c', alpha=0.3)
        plt.fill_between(x, ci_down_2, ci_up_2, color='c', alpha=0.2)
        plt.plot([x[1],x[1]], [target_after, target_expected], 'k-', \
          label='Intervention Effect ({}%)'.format(intervention_percent))
        #plt.axhline(y=avg_county_list_usages[feat], color='g', linestyle='-',label='Average {}'.format(feat))
        #plt.axhline(y=weighted_avg_county_list_usages[feat], color='g', linestyle='--',label='Weighted Average {}'.format(feat))
        plt.title("{}'s {} Before/After {}".format(fips_to_name[target], feat, event_name))

        # Format plot
        plt.xticks(rotation=45, ha='right')
        ax.set_xticks(xticks)
        ax.set_xticklabels([
          "{} Weeks Before".format(default_event_buffer + default_before_start_window + 1),
          "{} Week Before".format(default_event_buffer),
          "{} week After".format(default_event_buffer),
          "{} Weeks After".format(default_event_buffer + default_after_end_window + 1)
        ])
        plt.ylabel(str(feat))
        plt.legend()
        plt.tight_layout()

        plt_name = "covid_plots/did_{}_{}_before_after_covid_case.png".format(target, feat)

        plt.savefig(plt_name) # optionally save all figures

    # Aggregate Average Feature Usage
    # all_target_befores = list(target_befores.values())
    # all_target_diffs = list(target_diffs.values())
    # all_target_afters = all_target_befores + all_target_diffs
    # all_avg_matched_befores = list(avg_matched_befores.values())
    # all_avg_matched_diffs = list(avg_matched_diffs.values())
    # all_avg_matched_afters = list(avg_matched_afters.values()) # all_avg_matched_befores + all_avg_matched_diffs

    # # Calculate befores, afters, and diffs
    # target_before = np.mean(all_target_befores) # average of all targets before [scalar]
    # target_diff = np.mean(all_target_diffs) # average of all target diffs [scalar]
    # target_after = target_before + target_diff  # average of all target afters [scalar]

    # avg_match_before = np.mean(all_avg_matched_befores)
    # avg_match_diff = np.mean(all_avg_matched_diffs)
    # avg_match_after = np.mean(all_avg_matched_afters) # avg_match_before + avg_match_diff

    # target_expected = target_before + avg_match_diff  # average of all targets_before + avg_matched_diff [scalar]
    # intervention_effect = target_after - target_expected
    # intervention_percent = round(intervention_effect/target_expected * 100.0, 2)

    # std_match_before = np.std(all_avg_matched_befores)
    # std_match_after = np.std(all_avg_matched_afters)
    # std_match_diff = np.std(all_avg_matched_diffs)

    # Print out the results in sorted order
    print("\nSorted Results for COVID Case Changes:")
    for percent_change in sorted(percent_change_map.keys()):
      feature = percent_change_map[percent_change]
      stderr = stderr_change_map[feature]
      print("{} actual diff was {} stderr's larger than the counterfactual diff".format(feature, stderr))
    print("\nSummary for COVID Case Change:")
    for feat in list_features:
      feat_diffs = [key for key,value in percent_change_map.items() if value.split(":")[1] == feat]
      print("-> {} average diff was {} stderr's larger than the counterfactual diff".format(feat, np.mean(feat_diffs)))
    all_diffs = list(percent_change_map.keys())
    print("-> Overall average diff was {} stderr's larger than the counterfactual diff".format(np.mean(all_diffs)))
