'''Loads trained models and prints performance accuracy on training and validation data'''

import h2o

# Initialize h2o cluster
h2o.init(nthreads=-1,
         max_mem_size='6G')
h2o.remove_all()

# Load Models
path = '/Users/LiamRoberts/rossmann_retail/models/'

# Random Forest
forest_path = 'DRF_model_python_1543549695844_1'
rdf_model = h2o.load_model(f'{path}{forest_path}')

# XG Boost
xg_path = 'XGBoost_model_python_1543549100463_1'
xg_model = h2o.load_model(f'{path}{xg_path}')

# Check Performance
print('Random Forest Model')
print(rdf_model.model_performance(train=True))
print(rdf_model.model_performance(valid=True))

print('XGBoost Model')
print(xg_model.model_performance(train=True))
print(xg_model.model_performance(valid=True))

# Shutdown Cluster
h2o.cluster().shutdown()
