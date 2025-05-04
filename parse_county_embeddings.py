import pandas as pd

from pymysql import connect # type: ignore

print('Connecting to MySQL...')

# Open default connection
connection  = connect(read_default_file="~/.my.cnf")

#table = "ctlb2.feat$roberta_la_meL23nosent_wavg$ctlb_2020$county"
#table = "ctlb2.feat$roberta_la_meL23dlatk_avg$ctlb_2020$county"
table = "ctlb2.feat$roberta_la_meL23nosent_avg$ctlb_2020$county"

# SQL query to fetch the table data
query = """
SELECT id, group_id, feat, value, group_norm 
FROM ctlb2.feat$roberta_la_meL23nosent_wavg$ctlb_2020$county;
"""

# Load data into a Pandas DataFrame
df = pd.read_sql(query, connection)
print(df.head())

# Close the database connection
connection.close()

# Convert the DataFrame to wide format using pivot
wide_df = df.pivot(index='group_id', columns='feat', values='value')
wide_df.reset_index(inplace=True)
# drop index name
wide_df.index.name = None
print(wide_df.head())

# Save the wide format DataFrame to a CSV file
wide_df.to_csv(table + ".csv", index=False)

print("Wide format transformation completed!")