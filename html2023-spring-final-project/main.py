import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_df = pd.read_csv('submission.csv')

y = train_df[['Danceability']].to_numpy().squeeze()
use_list = ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo', 'Duration_ms']
X = train_df[use_list].to_numpy()
X_test = test_df[use_list].to_numpy()

# drop_list = ['Danceability', 'id']
# X = train_df.drop(drop_list, axis = 1).values

# print(y)
# print(X)
# print(X_test)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
predictions_array = []
CV_score_array    = []
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    regressor =  HistGradientBoostingRegressor(loss='absolute_error')
    regressor.fit(X_train, y_train)
    predictions_array.append(regressor.predict(X_test))
    CV_score_array.append(mean_absolute_error(y_valid,regressor.predict(X_valid)))
predictions = np.mean(predictions_array,axis=0)
predictions[predictions<0] = 0
predictions[predictions>9] = 9
print("The average CV mean absolute error is %d" % np.mean(CV_score_array,axis=0))
sample_df.iloc[:,1:] = np.rint(predictions)
sample_df.to_csv('submission.csv',index=False)
