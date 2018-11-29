'''Loads trained models and prints performance accuracy on training and validation data'''

import h2o

#Initialize h2o cluster
h2o.init(nthreads=-1,max_mem_size='6G')
h2o.remove_all()

#Load Model/s
forest_path ='/Users/liamroberts/Desktop/Datasets/Rossmann/Random_forest_model/DRF_model_python_1543508229496_1'
rdf_model = h2o.load_model(forest_path)

xg_path = '/Users/liamroberts/Desktop/Datasets/Rossmann/models/XGBoost_model_python_1543515884361_1'
xg_model = h2o.load_model(xg_path)

#Check Performance
print('Random Forest Model')
print(rdf_model.model_performance(train=True))
print(rdf_model.model_performance(valid =True))

print('XGBoost Model')
print(xg_model.model_performance(train=True))
print(xg_model.model_performance(valid =True))

#Shutdown Cluster
h2o.cluster().shutdown()
