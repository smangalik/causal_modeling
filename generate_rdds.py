"""
load the virtual environment: `source /data/smangalik/myenvs/diff_in_diff/bin/activate`
run as `python3.5 generate_rdds.py --covid_case`
run as `python3.5 generate_rdds.py --covid_death`
run as `python3.5 generate_rdds.py --worst_shooting`
run as `python3.5 generate_rdds.py --random`
"""

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

# Ignore warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Process feature table for RDD analysis")
parser.add_argument('--random', dest="random", default=False ,action='store_true', help='Assigns random valid events to each county')
parser.add_argument('--covid_case', dest="covid_case", default=False ,action='store_true', help='Evaluate the first COVID-19 case per county')
parser.add_argument('--covid_death', dest="covid_death", default=False ,action='store_true', help='Evaluate the first COVID-19 death per county')
parser.add_argument('--worst_shooting', dest="worst_shooting", default=False ,action='store_true', help='Evaluate the worst fatal shooting per county')
args = parser.parse_args()

# is the analysis being done on topics? (topic_num needs to be interpreted)
randomize_events = args.random

# Where to load data from
data_file = "/data/smangalik/lbmha_yw_cnty.csv" # from research repo
#data_file = "/data/smangalik/lbmha_yw_cnty_undifferenced.csv" # from research repo

# How many of the top populous counties we want to keep
top_county_count = 100 # 3232 is the maximum number, has a serious effect on results, usually use 1000

# user threshold (ut) required to consider a county
ut = 200 

# RDD Windows
default_before_start_window = 9 # additional weeks to consider before event start
default_after_end_window = 9    # additional weeks to consider after event end
default_event_buffer = 0        # number of weeks to ignore before and after event

# Confidence Interval Multiplier
ci_window = 1.96

# County-wise Socioeconomic Status
print("Loading SES Data...")
ses = pd.read_excel("/users2/smangalik/causal_modeling/LBMHA_Tract.xlsx")
ses['fips'] = ses['cfips'].astype(str).str.zfill(5)
ses = ses.groupby('fips').mean().reset_index()
ses = ses.dropna(subset=['fips','ses3'])
ses['ses3'] = ses['ses3'].astype(int)
print(ses[['fips','ses3']])

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

        
# Pick the events to use
if args.covid_case:
  print("Using First Covid Case Events")
  county_events = first_covid_case
elif args.covid_death:
  print("Using First Covid Death Events")
  county_events = first_covid_death
elif args.worst_shooting:
  print("Using Worst Shooting Events")
  county_events = worst_shooting
else:
  print("NO EVENT CHOSEN, defaulting to first covid case")
  county_events = first_covid_case

if randomize_events:
  # get unique event dates from the events
  possible_event_dates = list(set([event_start for event_start, _, _ in county_events.values()]))
  print("Randomizing Events with",len(possible_event_dates),"Possible Dates")
  import random
  for fips in county_events.keys():
    county_events[fips] = [random.choice(possible_event_dates),None,"Random Event"]
  print("Random Events: First Event for 40143",county_events.get('40143'))
else:
  print("Using Real Events: First Event for 40143",county_events.get('40143'))

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
    if county_feats.get(county,None) is not None:
      if county_feats[county].get(date,None) is not None: 
        if county_feats[county][date].get(feat,None) is not None:
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
      feat_usages.append(np.nan)
      
  return feat_usages

def regression_feat_usage_from_dates(county,dates,feat='Depression') -> float:
  # Get the feature usage for the county on the dates
  feat_usages = feat_usage_from_dates(county,dates,feat)

  # Report the slope and intercept of the date and feat usage
  feat_usages = np.array(feat_usages)
  mask = ~np.isnan(feat_usages) # drop NaNs
  X = np.arange(len(feat_usages))[mask].reshape(-1,1)
  y = feat_usages[mask]
  
  if len(X) == 0:
    return np.nan, np.nan
  
  regression = LinearRegression().fit(X, y)
  return regression.coef_[0], regression.intercept_


def regression_usage_before_and_after(county, feat, event_start, event_end=None,
                                 before_start_window=default_before_start_window,
                                 after_start_window=default_after_end_window,
                                 event_buffer=default_event_buffer) -> Tuple[ Tuple[float,float], Tuple[float,float], List[str], List[str] ]:
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

    # Require n_users >= UT for each county
    df = df[df['n_users'] >= ut]

    print("Data from CSV:", data_file)
    print(df.head(10))
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
    
    print("county_feats['40143']['2020_07']['Depression'] =",county_feats['40143']['2020_07']['Depression'])
    print("county_feats['40143']['2020_08']['Depression'] =",county_feats['40143']['2020_08']['Depression'])
    print("county_feats['40143']['2020_20']['Anxiety'] =",county_feats['40143']['2020_20']['Anxiety'])
    available_yws = list(county_feats['40143'].keys())
    available_yws.sort()
    print("\nAvailable weeks for 40143:",  available_yws)
    
    # Display nearest neighbors for first county
    #test_county = '36103' # Suffolk, NY
    test_county = '40143' # Washington, DC
    test_county_index = list(populous_counties).index(test_county)

    # Calculate Average and Weighted Average Weekly Feat Score for Baselines
    county_list = county_feats.keys() # all counties considered in orde
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


    # Calculate RDDs dict[county] = value
    target_befores = {}
    target_afters = {}
    target_diffs = {}
    
    percent_change_map = {}
    amount_change_map = {}

    output_dicts = []

    # Delete all previous plots
    print("Deleting all previous plots...")
    files = glob.glob('/users2/smangalik/causal_modeling/rdd_plots/*')
    for f in files:
      os.remove(f)
    
    target_counties = set(populous_counties)
    
    print("\nCalculating RDDs for",len(target_counties),"counties")
    for target in tqdm(target_counties):
      
      # Event for each county
      blank_event = [None,None,None]
      target_event_start, target_event_end, event_name = county_events.get(target,blank_event)

      # If no event was found for this county, skip
      if target_event_start is None and target_event_end is None:
        #print("Skipping {} ({}) due to missing event".format(target,fips_to_name.get(target)))
        continue # no event was found for this county
      
      #print()
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
        
        # RDD Calculation # TODO how are these calculated?
        intervention_effect = target_diff[1]
        intervention_percent = round(intervention_effect/target_before_intercept * 100.0, 2)
        is_significant = True # TODO: implement significance test

        # print("County: {} ({}) studying {} against {}".format(target, fips_to_name.get(target), feat, event_name))
        
        increase_decrease = "increased" if intervention_effect > 0 else "decreased"
        
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
        amount_change_map[intervention_effect] = [target,feat,middle_middle]

        # For writing to a dataframe
        pandas_entry = {
          "fips": target, # FIPS code of the county
          "county": fips_to_name.get(target), # Name of the county
          "feat": feat, # Either Depression or Anxiety
          "event_name": event_name, # Name of the event that we are measuring the feat before and after
          "event_date": target_event_start, # Date of the event
          "event_date_centered": middle_middle, # The date in the middle of the week containing the event_date   
          "target_before_slope": target_before[0], # feat score before the event
          "target_after_slope": target_after[0], # feat score after the event
          "target_before_intercept": target_before[1], # feat score before the event
          "target_after_intercept": target_after[1], # feat score after the event
          "target_diff_slope": target_diff[0], # feat score after - feat score before
          "target_diff_intercept": target_diff[1], # feat score after - feat score before
          "intervention_effect": intervention_effect, # feat score after - target_diff_expected
          "intervention_percent": intervention_percent, # intervention_effect / target_expected * 100
        }
        output_dicts.append(pandas_entry)

        # Only plot significant results
        if not is_significant:
          # print("NOT significant: County {} ({}) for {}, {} was not greater than {}\n --- ".format(
          #   feat,target,event_name,abs(intervention_effect),stddev_match_diff*ci_window)
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
        
        if randomize_events:
          continue # skip plotting for randomized events

        # --- Plotting ---

        # Calculate in-between dates and xticks
        x = np.array(range(1, len(dates_before) + len(dates_after)))
        xticklabels = [
            x_i - len(dates_before) for x_i in x
          ]
        xticklabels[len(dates_before)-1] = target_event_start.strftime('%Y-%m-%d')
        x_before = np.array(x[:len(dates_before)])
        x_after = np.array(x[len(dates_before)-1:])
        
        y_before = [ avg_county_list_usages[yw][feat] if yw in avg_county_list_usages else np.nan for yw in dates_before ]
        y_after =  [ avg_county_list_usages[yw][feat] if yw in avg_county_list_usages else np.nan for yw in dates_after ]
        y_vals = np.array(y_before + y_after[1:]) # skip the first repeat date in "after"
        y_before = np.array(y_before)
        y_after = np.array(y_after)

        # Create DiD Plot
        plt.clf() # reset old plot
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # Horizontally stacked plots
        fig.set_size_inches(10, 6)
        ax1.scatter(x[:len(dates_before)-1], y_vals[:len(dates_before)-1], label='Target (Before)')
        ax1.scatter(x[len(dates_before)-1], y_vals[len(dates_before)-1], color='black',label='Target (During)')
        ax1.scatter(x[len(dates_before):], y_vals[len(dates_before):], label='Target (After)')
        
        # print('x_before',x_before)
        # print('y_before',y_before)
        # print('x_after',x_after)
        # print('y_after',y_after)        
        
        # Plot vertical line for event
        ax1.axvline(x=len(dates_before), color='black', linestyle='--', label='Event')
        ax2.axvline(x=len(dates_before), color='black', linestyle='--', label='Event')
        
        # Plot line of best fit for Target (Before)
        mask = ~np.isnan(y_before)
        regression = LinearRegression().fit( x_before[mask].reshape(-1, 1), y_before[mask] )
        ax1.plot(x_before, regression.predict(x_before.reshape(-1, 1)), color='blue', linestyle='--', alpha=0.3, label='Target (Before) Fit')
        
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
        
        #plt.axhline(y=avg_county_list_usages[feat], color='g', linestyle='-',label='Average {}'.format(feat))
        #plt.axhline(y=weighted_avg_county_list_usages[feat], color='g', linestyle='--',label='Weighted Average {}'.format(feat))
        fig.suptitle('Intervention Effect (Slope: {} Intercept: {})'.format(round(target_diff[0],3), round(target_diff[1],3)))
        
        # Plot the average and weighted average per week
        
        ax2.scatter(x, y_vals, color='green', label='All', alpha=0.8)

        # Plot the target county per week
        # x_pos = np.arange(1,8)
        # y_pos = [begin_before, end_before, middle_before, middle_middle, middle_after, begin_after, end_after]
        # x_pos = np.arange(1,8)
        # y_vals = [county_feats.get(target,{}).get(date_to_yearweek(yw),{}).get(feat,np.nan) for yw in y_pos]
        #ax2.plot(x_pos, y_vals, 'b-', label='Target {}'.format(feat), alpha=0.3)

        # Plot the matched counties per week
        # x_pos = np.arange(1,8)
        # y_vals = np.zeros((len(matched_counties[target][feat]),len(y_pos)))
        # for i, matched_county in enumerate(matched_counties[target][feat]):
        #   y_vals[i] = [county_feats.get(matched_county,{}).get(date_to_yearweek(yw),{}).get(feat,np.nan) for yw in y_pos]
        # y_vals = np.nanmean(y_vals,axis=0)
        #ax2.plot(x_pos, y_vals, 'r-', label='Matched {}'.format(feat), alpha=0.3)

        ymin = min([min([y for y in y_vals if y is not None])])
        ymax = max([max([y for y in y_vals if y is not None])])

        # Format plot
        plt.ylabel("County {} Score".format(feat))
        for ax in [ax1, ax2]:
          ax.set_xticks(x)
          ax.set_xticklabels(xticklabels, rotation=80)
          ax.axis(ymin=ymin, ymax=ymax)
          ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt_name = "rdd_plots/rdd_{}_{}_before_after_event.png".format(target, feat)

        plt.savefig(plt_name) # optionally save all figures
        
    # Print out the results in sorted order
    print("\nSorted Results for Event Changes:")
    for percent_change in sorted(percent_change_map.keys()):
      feature = percent_change_map[percent_change]
    print("\nSummary for Event Change:")
    for feat in list_features:
      feat_diffs = [key for key,value in percent_change_map.items() if value.split(":")[1] == feat]
      print("-> {} average diff was {} stddev's larger than the counterfactual diff".format(feat, np.mean(feat_diffs)))
    all_diffs = list(percent_change_map.keys())
    print("-> Overall average diff was {} stddev's larger than the counterfactual diff".format(np.mean(all_diffs)))
    
    # Create a pandas dataframe
    print("\nCreating Pandas Dataframe of Outcomes")
    columns = ['fips','county','feat','event_name','event_date','event_date_centered','target_before_slope','target_after_slope','target_before_intercept','target_after_intercept','target_diff_slope','target_diff_intercept','intervention_effect','intervention_percent']
    output = pd.DataFrame(output_dicts)[columns]
    print(output.sort_values(by=['fips','feat']))
    
    # Merge the SES data
    output = output.merge(ses, how='left', on='fips')
    output_ses_1 = output[output['ses3'] == 1]
    output_ses_2 = output[output['ses3'] == 2]
    output_ses_3 = output[output['ses3'] == 3]
    
    def print_outputs(output_df):
      output_df = output_df.copy(deep=True)
      outcomes  = ['intervention_effect','intervention_percent']
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
      outcome = 'intervention_effect'
      dat = outcomes_feat[outcome].replace([np.inf, -np.inf], np.nan).dropna()
      for col in corr_columns:
        corr = np.corrcoef(dat,output[col].loc[dat.index])[0,1]
        print("-> Correlation between {} and {} = {}".format(outcome,col,round(corr,4)))
    
    # Plot amount change over time if not doing a randomization
    if not randomize_events:
      
      # Write the output to a CSV
      output.to_csv("rdd_data.csv",index=False)
      
      # Plot the amount change over time
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
        plt_name = "over_time_changes/amount_change_{}_before_after_event.png".format(feat)
        plt.savefig(plt_name)
        print("Saved",plt_name)