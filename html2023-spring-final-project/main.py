import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

train_df = pd.read_csv('train.csv')
train_df = train_df.fillna(0)
test_df = pd.read_csv('test.csv')
test_df = test_df.fillna(0)
sample_df = pd.read_csv('submission.csv')

y = train_df[['Danceability']].to_numpy().squeeze()
use_list = ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream',
            'Comments']
X = train_df[use_list].to_numpy()
X_test = test_df[use_list].to_numpy()

poly = PolynomialFeatures(3)
poly.fit_transform(X)

kf = KFold(n_splits=10, random_state=42, shuffle=True)
predictions_array = []
CV_score_array    = []
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    regressor = HistGradientBoostingRegressor()
    # regressor = LinearRegression()
    # regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    predictions_array.append(regressor.predict(X_test))
    CV_score_array.append(mean_absolute_error(y_valid,regressor.predict(X_valid)))
predictions = np.mean(predictions_array,axis=0)
predictions[predictions<0] = 0
predictions[predictions>9] = 9
print(f"CV scaore {CV_score_array}")
sample_df['Danceability'] = np.rint(predictions)
sample_df.to_csv('submission.csv',index=False)
