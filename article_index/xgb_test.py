import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
from glob import glob
from copy import deepcopy
import yaml

#all_data = pd.read_pickle('article_index/data/all_data_reindex.pkl')

data_sample = all_data.sample(frac=0.1)

X = data_sample[yaml_feature_names]
print("X shape:")
print(X.shape)
print("      ")
xgb.fit(X, y)
score = xgb.score(X, y)

print("score:")
print(score)
