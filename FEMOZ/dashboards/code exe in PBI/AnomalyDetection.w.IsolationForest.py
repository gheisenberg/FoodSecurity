from   sklearn.preprocessing import StandardScaler
from   sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

cpi_data = pd.read_csv("C:\Users\heisenberg\Desktop\data\THKoeln\Research\GIT\FoodSecurity\trunk\FEMOZ\scripts\data\CPI_FAOSTAT_data_7-11-2022.csv")

cpi_data = cpi_data.loc[cpi_data['Area'] == "Mozambique"]     
cpi_data = cpi_data.loc[cpi_data['Item Code'] == 23014]     
cpi_data = cpi_data.reset_index()

dates = pd.Series(pd.date_range(start='2001-01', end='2022-1', freq='M', normalize=True))

# add it to the dataframe
cpi_data['Dates'] = dates

def create_IsolationForest_model(df_passed_in):
    data = df_passed_in[['Value']]
    
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    
    # see documentation for learning about the IF parameters
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    model = IsolationForest(max_samples=3500, random_state = 1, contamination= 'auto')
    
    model.fit(data)
    return model

def IsolationForest_prediction(df_passed_in, model):
    
    data = df_passed_in[['Value']]
    
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    
    data = pd.DataFrame(np_scaled)
    df_passed_in['anomaly_IF'] = list(model.predict(data))
    
    return df_passed_in

# create the IF model for the specific price data
model = create_IsolationForest_model(cpi_data)
# Now, by means of the IF model we can look for the anomalies
cpi_data_with_anomalies = IsolationForest_prediction(cpi_data, model)
cpi_data_with_anomalies['anomaly'] = np.where(cpi_data_with_anomalies['anomaly_IF']==1, False, True)
cpi_data_with_anomalies.drop(['anomaly_IF'], inplace = True, axis=1)