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
from sklearn.preprocessing import PolynomialFeatures

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_df = pd.read_csv('submission.csv')

# train_df = train_df.fillna(0)
# test_df = test_df.fillna(0)

train_df['Album_type'].replace('album', 3, inplace=True)
train_df['Album_type'].replace('single', 2, inplace=True)
train_df['Album_type'].replace('compilation', 1, inplace=True)
test_df['Album_type'].replace('album', 3, inplace=True)
test_df['Album_type'].replace('single', 2, inplace=True)
test_df['Album_type'].replace('compilation', 1, inplace=True)

y = train_df[['Danceability']].to_numpy().squeeze()
use_list = ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream',
            'Comments', 'Licensed', 'official_video', 'Album_type']
# use_list.remove()
X = train_df[use_list].to_numpy()
X_test = test_df[use_list].to_numpy()

# poly = PolynomialFeatures(2)
# X = poly.fit_transform(X)
# X_test = poly.fit_transform(X_test)

kf = KFold(n_splits=10, random_state=42, shuffle=True)
predictions_array = []
CV_score_array    = []
cnt = 0
for train_index, valid_index in kf.split(X):
    cnt += 1
    print(cnt)
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    regressor = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, max_iter=400, max_leaf_nodes=200)
    # regressor = tree.DecisionTreeRegressor()
    # regressor = LinearRegression()
    # regressor = RandomForestRegressor()
    # regressor = AdaBoostRegressor()
    regressor.fit(X_train, y_train)
    predictions_array.append(regressor.predict(X_test))
    CV_score_array.append(mean_absolute_error(y_valid,regressor.predict(X_valid)))
predictions = np.mean(predictions_array,axis=0)
predictions[predictions<0] = 0
predictions[predictions>9] = 9
print(f"CV score {np.mean(CV_score_array)}")
sample_df['Danceability'] = np.rint(predictions)
sample_df.to_csv('submission.csv',index=False)
