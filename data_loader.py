'''Preprocesses and cleans base kaggle data then saves it to a csv in the local directory'''

import pandas as pd

# Specify Working Directory
directory = '/Users/LiamRoberts/Desktop/Professional/DataScience/Rossmann Retail Forecasting/'

# Import Data
store = pd.read_csv(f'{directory}store.csv',
                    low_memory=False)

state = pd.read_csv(f'{directory}store_states.csv',
                    low_memory=False)

df = pd.read_csv(f'{directory}train.csv',
                 low_memory=False,
                 parse_dates=True,
                 index_col='Date')

# Encode Data Times
df['Month'] = df.index.month
df['WeekOfYear'] = df.index.weekofyear

# Merge Data Frames
df = pd.merge(df,
              store,
              on = 'Store')

df = pd.merge(df,
              state,
              on = 'Store')

# Remove days where sales are zero and when store is Closed
df = df[(df['Sales'] > 0)]
df = df[(df['Open'] == 1)]

# -----Feature Engineering------ #

# Store Specific Metrics
agg_df = df.groupby(by='Store').mean()
agg_df['SalesPerCustomer'] = agg_df['Sales']/agg_df['Customers']
agg_df['MeanStoreSales'] = agg_df['Sales']
agg_df['MeanStoreCustomers'] = agg_df['Customers']
agg_df = agg_df[['MeanStoreSales',
                 'MeanStoreCustomers',
                 'SalesPerCustomer']]
df = pd.merge(df,
              agg_df,
              on='Store')

# Promo Sales Ratio
promo = df[df['Promo'] == 1].groupby('Store').mean()
nopromo = df[df['Promo'] == 0].groupby('Store').mean()
promo['PromoRatio'] = promo['Sales']/nopromo['Sales']
promo['A'] = promo['Sales']
promo = promo[['PromoRatio',
               'A']]

df = pd.merge(df,
              promo,
              on='Store')

# Promo Days Ratio
promo = df[df['Promo'] == 1].groupby('Store').sum()
nopromo = df[df['Promo'] == 0].groupby('Store').sum()
promo['PromoDaysRatio'] = promo['Open']/nopromo['Open']
promo['B'] = promo['Sales']
promo = promo[['PromoDaysRatio',
               'B']]

df = pd.merge(df,
              promo,
              on='Store')

df.drop(columns=['A',
                 'B',
                 'Promo2SinceWeek',
                 'Promo2SinceYear'],
        inplace=True)

# ------Save Data to Csv------ #

# Train test split ~90% through data
split_point = int(len(df)*0.9)
train = df[:split_point]
test = df[split_point:]

# Save train and test to local folder
train.to_csv('train.csv',
             index=False)
test.to_csv('test.csv',
            index=False)
print("~Files Succesfully Saved~")
