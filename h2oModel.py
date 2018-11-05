#A sample pipeline using a H2ORandomForestEstimator in python for a kaggle data set, available:
#https://www.kaggle.com/c/rossmann-store-sales

#For a guide on setting up an H2o enviornment in anaconda check out:
#https://liamwoodroberts.com/2018/11/01/setting-up-an-h20-environment-with-anaconda-mac/

#All Packages Used
import pandas as pd
import h2o
import numpy as np
from h2o.estimators.random_forest import H2ORandomForestEstimator

#Import Data
test = pd.read_csv('test.csv',parse_dates=True,index_col='Date')
data = pd.read_csv('train.csv',parse_dates=True,index_col='Date')
store = pd.read_csv('store.csv')

#Convert Date Columns
def add_dates(data):
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['WeekOfYear'] = data.index.weekofyear
    return data
test = add_dates(test)
data = add_dates(data)

#Fill NaN values for store
store.fillna(value = 0,inplace = True)

#Merge training dataframe
pd_df = data.merge(store,on='Store')

#Merge Testing Dataframe
test_df = test.merge(store,on='Store')

#Make sure test data is in order for prediction
test_df.set_index('Id',inplace = True)
test_df.sort_index(inplace=True)

#Kaggle doesnt count days where store was not open in its scoring
pd_df = pd_df[(pd_df.Sales>0)]

#Initialize h2o cluster and clear any previously saved info from the cluster
h2o.init()
h2o.remove_all()

#Create h2o frame for training models
train = h2o.H2OFrame(python_obj=pd_df)
testh2o = h2o.H2OFrame(python_obj=test_df)

#Log transform sales data and assign X and y columns names
train['log_sales'] = train['Sales'].log()
X_labels = [i for i in train.col_names if (
            i not in ['Sales','Customers','log_sales'])]
y_labels = 'log_sales'

#Create and Train Model
rf = H2ORandomForestEstimator(
    ntrees=100,
    max_depth = 30,
    stopping_rounds = 5,
    stopping_tolerance = 1e-4
    )

#h2o supports .fit() to fit into the sklearn pipeline however recommends using
#.train() and passing the full dataframe into the model
rf.train(x=X_labels,y=y_labels,training_frame=train)

#Get predictions and transfrom them back to untransformed state
predictions = rf.predict(testh2o)
pred = predictions.expm1()

#Convert h2o frame back into a pandas data frame for data manipulation
#and CSV conversion
submission = pred.as_data_frame()

#Convert Data Frame to CSV for submission
submission['Id'] = test['Id'].values
submission['Sales'] = submission['expm1(predict)']
submission.set_index('Id',inplace = True)
submission.drop(columns = 'expm1(predict)',inplace = True)
submission.to_csv('h2osub.csv')

#This line will prevent the cluster from getting its memory too filled up
#if the notebook is run multiple times
h2o.cluster().shutdown()
