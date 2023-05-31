import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_df = pd.read_csv('submission.csv')

# train_df = train_df.fillna(0.0)
# test_df = test_df.fillna(0.0)

str_list = ['Album_type', 'Composer', 'Channel', 'Artist', 'Track', 'Album','Title']

for col in str_list:
    tmp = []
    for s in train_df[col]:
        if type(s) == float:
            tmp.append(s)
        else:
            tmp.append(len(s))
    train_df[col] = np.array(tmp)
    tmp = []
    for s in test_df[col]:
        if type(s) == float:
            tmp.append(s)
        else:
            tmp.append(len(s))
    test_df[col] = np.array(tmp)

y = train_df[['Danceability']].to_numpy().squeeze()
use_list = ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream',
            'Comments', 'Licensed', 'official_video', 'Album_type', 'Track',
            'Album',
            'Title', 'Channel', 'Composer', 'Artist']

X = train_df[use_list].to_numpy()
X_test = test_df[use_list].to_numpy()

# X = np.concatenate((train_df[use_list].to_numpy(),np.array(type_X).T),axis=1)
# X_test = np.concatenate((test_df[use_list].to_numpy(),np.array(type_X_test).T),axis=1)

# scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
# X = scaler.transform(X)

# scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_test)
# X_test = scaler.transform(X_test)

# poly = PolynomialFeatures(2)
# X = poly.fit_transform(X)
# X_test = poly.fit_transform(X_test)

# param_grid = {
#     'loss': ('squared_error', 'absolute_error', 'poisson', 'quantile'),
#     'learning_rate': (0.01, 0.1, 1, 10),
#     'max_iter': (100, 200, 400),
#     'max_leaf_nodes': (3, 10, 30, 100, 200 ,400),
#     'min_samples_leaf': (10, 20, 30),
#     }
# regressor = HistGradientBoostingRegressor()
# grid_search = GridSearchCV(regressor, param_grid = param_grid)
# grid_search.fit(X, y)
# print(f"The best set of parameters is: {grid_search.best_params_}")

# The best set of parameters is: {'learning_rate': 0.01, 'loss': 'squared_error', 'max_iter': 400, 'max_leaf_nodes': 100, 'min_samples_leaf': 30}

# kf = KFold(n_splits=10, random_state=42, shuffle=True)
predictions_array = []
# CV_score_array    = []
# cnt = 0
# for train_index, valid_index in kf.split(X):
#     cnt += 1
#     print(cnt)
#     X_train, X_valid = X[train_index], X[valid_index]
#     y_train, y_valid = y[train_index], y[valid_index]
#     regressor = HistGradientBoostingRegressor(loss = 'absolute_error', learning_rate = 0.04, max_iter = 400,
#                                             max_leaf_nodes = 100, min_samples_leaf = 30)
#     # regressor = tree.DecisionTreeRegressor()
#     # regressor = LinearRegression()
#     # regressor = RandomForestRegressor()
#     # regressor = AdaBoostRegressor()
#     regressor.fit(X_train, y_train)
#     predictions_array.append(regressor.predict(X_test))
#     CV_score_array.append(mean_absolute_error(y_valid,regressor.predict(X_valid)))
# predictions = np.mean(predictions_array,axis=0)
# predictions[predictions<0] = 0
# predictions[predictions>9] = 9
# print(f"CV score {np.mean(CV_score_array)}")
# sample_df['Danceability'] = np.rint(predictions)
# sample_df.to_csv('submission.csv',index=False)


regressor = HistGradientBoostingRegressor(loss = 'absolute_error', learning_rate = 0.04, max_iter = 400,
                                            max_leaf_nodes = 100, min_samples_leaf = 30)
regressor.fit(X, y)
predictions_array.append(regressor.predict(X_test))
predictions = np.mean(predictions_array,axis=0)
predictions[predictions<0] = 0
predictions[predictions>9] = 9
sample_df['Danceability'] = np.rint(predictions)
sample_df.to_csv('submission.csv',index=False)