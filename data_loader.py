import pandas as pd

#Set Root Directory
directory = '/Users/liamroberts/Desktop/Datasets/Rossmann/'

#Import Data
store = pd.read_csv(f'{directory}store.csv',
                    low_memory=False)

df = pd.read_csv(f'{directory}train.csv',
                 low_memory=False,
                 parse_dates=True,
                 index_col='Date')

#Encode Data Times
df['Month'] = df.index.month
df['WeekOfYear'] = df.index.weekofyear

#Merge Data Frames
df = pd.merge(df,store,on = 'Store')

#Remove Days Where Sales are Zero
df = df[df['Sales']>0]

#Train test split ~90% through data
split_point = int(len(df)*0.9)
train = df[:split_point]
test = df[split_point:]

#Save train and test to local folder
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)