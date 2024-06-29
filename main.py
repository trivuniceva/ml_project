import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


data_path = 'data/metadata.csv'
data = pd.read_csv(data_path)

# numerical_columns = data.select_dtypes(include=['number']).columns
# data_columns = data.columns

# print('len data columns', len(data_columns))
# X = data[numerical_columns]
# X = data[data_columns]
# print(X)
# print("len: ", len(X))

# num_rows_with_nan = X.isnull().sum()
# print(num_rows_with_nan)


# X = X.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark', 'musicMeta.musicAlbum'])
data = data.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark', 'musicMeta.musicAlbum'])
# num_rows_with_nan = X.isnull().sum()
# print(num_rows_with_nan)

scaler = MinMaxScaler()
data[['diggCount', 'shareCount', 'playCount', 'commentCount']] = scaler.fit_transform(data[['diggCount', 'shareCount', 'playCount', 'commentCount']])

# Kreiranje dodatnih karakteristika
data['likes_per_view'] = data['diggCount'] / data['playCount']
data['comments_per_view'] = data['commentCount'] / data['playCount']
data['shares_per_view'] = data['shareCount'] / data['playCount']

data.dropna(inplace=True)

X = data[['diggCount', 'shareCount', 'commentCount', 'likes_per_view', 'comments_per_view', 'shares_per_view']]
y = data['playCount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = PendingDeprecationWarnin = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_

# y_pred = model.predict(X_test)
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Primer kako koristiti model za predikciju broja pregleda
new_video_features = {
    'diggCount': 500,
    'shareCount': 100,
    'commentCount': 50,
    'likes_per_view': 0.05,
    'comments_per_view': 0.01,
    'shares_per_view': 0.02
}

new_video_df = pd.DataFrame([new_video_features])
predicted_views = best_model.predict(new_video_df)
print(f'Predicted number of views: {predicted_views[0]}')
original_views = scaler.inverse_transform(predicted_views.reshape(-1, 1))
print(f'Predicted number of views (original scale): {original_views[0][0]}')
