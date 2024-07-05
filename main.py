import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


def read_clean_data():
    print("data...")
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

    # data.dropna(inplace=True)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].applymap(lambda x: np.nan if (isinstance(x, float) and x > 1e10) else x)

    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    data[data.select_dtypes(include=[np.number]).columns] = data_imputed


    # numerical_columns = data.select_dtypes(include=[np.number]).columns
    # correlation_matrix = data[numerical_columns].corr()

    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Matrix')
    # plt.show()

    return data


def preprocessing(data):
    # numerical_columns = ['diggCount', 'shareCount', 'playCount', 'commentCount']

    # numerical_features = numerical_columns + ['likes_per_view', 'comments_per_view', 'shares_per_view']
    # categorical_features = ['authorMeta.verified', 'musicMeta.musicOriginal', 'downloaded']
    # textual_features = ['text']

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', MinMaxScaler(), numerical_features),
    #         ('cat', OneHotEncoder(), categorical_features),
    #         ('text', TfidfVectorizer(max_features=1000), 'text')  # TF-IDF vectorizer for text data
    #     ])

    # X = data[numerical_features + categorical_features + textual_features]
    # y = data['playCount']

    # X = data[['diggCount', 'shareCount', 'commentCount', 'likes_per_view', 'comments_per_view', 'shares_per_view']]
    
    features = ['authorMeta.fans', 'authorMeta.heart', 'diggCount', 'shareCount', 'commentCount']
    X = data[features]
    y = data['playCount']

    # numerical_features = ['authorMeta.fans', 'authorMeta.heart', 'diggCount', 'shareCount', 'commentCount', 'likes_per_view', 'comments_per_view', 'shares_per_view']
    # categorical_features = ['authorMeta.verified', 'musicMeta.musicOriginal', 'downloaded']
    # textual_features = ['text']

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', MinMaxScaler(), numerical_features),
    #         ('cat', OneHotEncoder(), categorical_features),
    #         ('text', TfidfVectorizer(max_features=1000), 'text')  # TF-IDF vectorizer for text data
    #     ])

    # X = data[numerical_features + categorical_features + textual_features]
    # y = data['playCount']

    return X, y


def train_data(data):
    print("traun")

    X, y = preprocessing(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = PendingDeprecationWarnin = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # X_transformed = preprocessor.fit_transform(X)
    # correlation_matrix = pd.DataFrame(X_transformed.toarray()).corr()

    # Plot the correlation matrix
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Matrix')
    # plt.show()

    model = RandomForestRegressor(random_state=42)
    # model.fit(X_train, y_train)

    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    
    # best_model = grid_search.best_estimator_

    # y_pred = best_model.predict(X_test)

    print("random.....")

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,
                                   cv=5, n_jobs=-1, random_state=42, verbose=1)

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    return y_test, y_pred


def evaluation(y_test, y_pred):
    print("evaluation")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}')

    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, alpha=0.3)
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    # plt.xlabel('Actual values')
    # plt.ylabel('Predicted values')
    # plt.title('Actual vs Predicted Values')
    # plt.show()

def new_video_predict():
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
    # print(f'Predicted number of views: {predicted_views[0]}')
    original_views = scaler.inverse_transform(predicted_views.reshape(-1, 1))
    # print(f'Predicted number of views (original scale): {original_views[0][0]}')


def main():
    data = read_clean_data()
    y_test, y_pred = train_data(data)
    evaluation(y_test, y_pred)

main()


