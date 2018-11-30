'''Trains and saves an XGBoost Model to local directory'''

import pandas as pd
import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator

#Set File Paths
data_directory = '/Users/LiamRoberts/Desktop/Professional/DataScience/Rossmann Retail Forecasting/'
save_folder = '/Users/LiamRoberts/rossmann_retail/models'

#Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Initialize h2o cluster
h2o.init(nthreads=-1,max_mem_size='6G',enable_assertions=False)
h2o.remove_all()

#Convert DataFrames to h2o frames
train = h2o.H2OFrame(python_obj=train)
test = h2o.H2OFrame(python_obj=test)

#Encode Categorical Variables
categorical = ['Store','DayOfWeek','Month','WeekOfYear']

for label in categorical:
    train[label] = train[label].asfactor()
    test[label] = test[label].asfactor()

#Log transform sales
train['log_sales'] = train['Sales'].log()
test['log_sales'] = test['Sales'].log()

#Assign X and y columns to pass into h2o train method
X_labels = [i for i in train.col_names if (i not in ['Sales','Customers','log_sales'])]
y_labels = 'log_sales'

#Create Model Using Default XGBoost Estimator
model = H2OXGBoostEstimator(seed =1)

model.train(x=X_labels,
            y=y_labels,
            training_frame=train,
            validation_frame=test)

#Save Model
model_path = h2o.save_model(model = model,
               path = f'{save_folder}',
               force = True)

print(model_path)

#Shutdown Cluster
h2o.cluster().shutdown()
