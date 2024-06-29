import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score


data_path = 'data/metadata.csv'
data = pd.read_csv(data_path)

numerical_columns = data.select_dtypes(include=['number']).columns
data_columns = data.columns

# print('len data columns', len(data_columns))
# X = data[numerical_columns]
X = data[data_columns]
# print(X)
# print("len: ", len(X))

num_rows_with_nan = X.isnull().sum()
# print(num_rows_with_nan)


X = X.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark', 'musicMeta.musicAlbum'])
num_rows_with_nan = X.isnull().sum()
# print(num_rows_with_nan)

scaler = MinMaxScaler()
data[['diggCount', 'shareCount', 'playCount', 'commentCount']] = scaler.fit_transform(data[['diggCount', 'shareCount', 'playCount', 'commentCount']])

# Kreiranje dodatnih karakteristika
data['likes_per_view'] = data['diggCount'] / data['playCount']
data['comments_per_view'] = data['commentCount'] / data['playCount']
data['shares_per_view'] = data['shareCount'] / data['playCount']

print(data.isnull().sum)
data.dropna(inplace=True)

# TODO: popuni vrednosti SimpleImputer opcija2

data.dropna(inplace=True)

