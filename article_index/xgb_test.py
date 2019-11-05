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

yaml_feature_names = [
    "proportion_internal_index_subscriber_read",
    "proportion_internal_article_subscriber_read",
    "proportion_direct_subscriber_read",
    "proportion_desktop_web_subscriber_read",
    "proportion_mobile_web_subscriber_read",
    "proportion_other_web_subscriber_read", "proportion_apps_subscriber_read",
    "unique_commenters", "comments", "save_adds_subscriber_read",
    "avg_session_hit_count_subscriber_read",
    "avg_session_read_count_subscriber_read", "words", "sentences",
    "cs_section_category", "section_category", "num_characters", "age_hours",
    "day_of_week"
]

xgb = XGBRegressor(
    max_depth=3,
    seed=2012,
    learning_rate=0.1,
    colsample_bylevel=0.9,
    subsample=0.7,
    n_estimators=1000,
    silent=True,
    objective='reg:linear',
    n_jobs=-1,
    min_child_weight=5,
    reg_lambda=3,
    gamma=2,
    colsample_bytree=0.9)

all_data = pd.read_pickle('all_data_reindex.pkl')

data_sample = all_data.sample(frac=0.1)

X = data_sample[yaml_feature_names]
print("X shape:")
print(X.shape)
print("      ")
xgb.fit(X, y)
score = xgb.score(X, y)

print("score:")
print(score)
