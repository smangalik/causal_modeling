"""
load the virtual environment: `source /data/smangalik/myenvs/diff_in_diff/bin/activate`
run as `python3.5 examine_diff_in_diffs.py`
"""

import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Get all CSV files starting with "diff_in_diff"
files = glob.glob('diff_in_diff*.csv')

# For each event type, load the data and examine the features
for file in files:
    full_df = pd.read_csv(file)
    print('\n\n',file, list(full_df.columns))
    print(full_df.head())
    
    event_name = file.replace('diff_in_diff_data_','').replace('.csv','')
    
    # Iterate over each feature
    for feature in full_df['feat'].unique():
        
        df = full_df[full_df['feat']==feature]
        print('*',feature)
        
        print("Number of rows:\t\t", len(df))
        print("Count of significant:\t", len(df[df['significant']==True]))  # noqa: E712
        print("Mean (Std) of intervention_effect:\t", df['intervention_effect'].mean(), '(', df['intervention_effect'].std(), ')')
        print("Mean (Std) of intervention_percent:\t", df['intervention_percent'].mean(), '(', df['intervention_percent'].std(), ')')
        print("Mean (Std) of neighbor_count:\t\t", df['neighbor_count'].mean(), '(', df['neighbor_count'].std(), ')')
        # Create a KDE plot of intervention_effect
        plt.clf()
        sns.kdeplot(df['intervention_effect'])
        plt.axvline(df['intervention_effect'].mean(), color='r', label='Mean ({})'.format(df['intervention_effect'].mean()))
        plt.title('Intervention Effect of {} on {}'.format(event_name,feature))
        plt.legend()
        plt.savefig("kde_plots/intervention_effect_{}_{}".format(feature, event_name))
        
        print("-"*25,'Only significant findings', "-"*25)
        df = df[df['significant']==True]  # noqa: E712
        print("Number of rows:\t\t", len(df))
        print("Mean (Std) of intervention_effect:\t", df['intervention_effect'].mean(), '(', df['intervention_effect'].std(), ')')
        print("Mean (Std) of intervention_percent:\t", df['intervention_percent'].mean(), '(', df['intervention_percent'].std(), ')')
        print("Mean (Std) of neighbor_count:\t\t", df['neighbor_count'].mean(), '(', df['neighbor_count'].std(), ')')
        # Create a KDE plot of intervention_effect
        plt.clf()
        sns.kdeplot(df['intervention_effect'])
        plt.axvline(df['intervention_effect'].mean(), color='r', label='Mean ({})'.format(df['intervention_effect'].mean()))
        plt.title('Intervention Effect of Sig. {} on {}'.format(event_name,feature))
        plt.legend()
        plt.savefig("kde_plots/intervention_effect_sig_{}_{}".format(feature, event_name))
    