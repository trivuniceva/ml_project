import pandas as pd
import numpy as np 


data_path = 'data/metadata.csv'
data = pd.read_csv(data_path)

numerical_columns = data.select_dtypes(include=['number']).columns
data_columns = data.columns

print('len data columns', len(data_columns))
# X = data[numerical_columns]
X = data[data_columns]
print(X)
print("len: ", len(X))

num_rows_with_nan = X.isnull().sum()
print(num_rows_with_nan)



X = X.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark', 'musicMeta.musicAlbum'])
num_rows_with_nan = X.isnull().sum()
print(num_rows_with_nan)




