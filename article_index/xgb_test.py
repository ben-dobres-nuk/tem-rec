import pandas as pd
import numpy as np
import random

from xgboost import XGBRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import cross_validate

import argparse
import logging
import warnings

logging.basicConfig(
    filename='seed_test_' + str(random.getrandbits(20)) + '.log',
    level=logging.DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-s',
    '--sample',
    type=float,
    default=0.001,
    help="sample size as a fraction")

args = parser.parse_args()

warnings.simplefilter(action='ignore', category=FutureWarning)

# removing unique commenters as a feature gives us a small improvement
# on seed variance with almost no impact on accuracy

yaml_feature_names = [
    "proportion_internal_index_subscriber_read",
    "proportion_internal_article_subscriber_read",
    "proportion_direct_subscriber_read",
    "proportion_desktop_web_subscriber_read",
    "proportion_mobile_web_subscriber_read",
    "proportion_other_web_subscriber_read",
    "proportion_apps_subscriber_read",

    # "unique_commenters",
    "comments",
    "save_adds_subscriber_read",
    "avg_session_hit_count_subscriber_read",
    "avg_session_read_count_subscriber_read",
    "words",
    "sentences",
    "cs_section_category",
    "section_category",
    "num_characters",
    "age_hours",
    "day_of_week"
]

# sample_frac = 0.001
all_data = pd.read_pickle('all_data_reindex.pkl')
data_sample = all_data.sample(frac=args.sample, random_state=801)

X = data_sample[yaml_feature_names]
y = data_sample['avg_dwell_time_subscriber_read']

# generate seeds for random sample later
# seed2 = random.sample(range(1, X.shape - 1), 1000)

# booster: gbtree, gblinear or dart?

xgb_params = {
    'max_depth': 2,
    'learning_rate': 0.1,
    'colsample_bylevel': 0.9,
    'subsample': 0.7,
    'n_estimators': 1000,
    'silent': True,
    # switched reg:linear for reg:squarederror
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'min_child_weight': 5,
    'reg_lambda': 3,
    'gamma': 10,
    'colsample_bytree': 0.9
}

# linsvr_params = {
#     'C': 1.0,
#     'dual': True,
#     'epsilon': 0.0,
#     'fit_intercept': True,
#     'intercept_scaling': 1.0,
#     'loss': 'epsilon_insensitive',
#     'max_iter': 1000000
# }
# svr_params = {'kernel': 'rbf', 'gamma': '0.1'}
# krr_params = {'kernel': 'rbf', 'gamma': '0.1'}


def trial_seed(model, params, n, seed_param):

    params[seed_param] = n

    model = model(**params)

    model.fit(X, y)
    # cross validate
    cv = cross_validate(model, X, y, cv=2)

    return ((model.predict(X), cv['test_score']))


def make_seed_dataset(model, params, n_seeds, seed_param):
    logging.info("model: {}".format(str(model)))
    logging.info("sample size: {}".format(len(data_sample.index)))
    logging.info(yaml_feature_names)
    logging.info(params)
    logging.info("Number of seeds: {}".format(n_seeds))
    trials = {"seeds": [], "test_score": []}
    for n in range(1, n_seeds):
        trial = trial_seed(model, params, n, seed_param)
        trials["seeds"].append(trial[0])
        trials["test_score"].append(trial[1][0])

    mean_cv_score = np.mean(trials["test_score"])

    df = pd.DataFrame(
        np.vstack(trials["seeds"]),
        columns=["row" + str(x) for x in range(1, y.shape[0] + 1)])
    logging.info("Created seed dataset of shape {}".format(df.shape))
    # Create summary dataframe with a row for each observation
    stats = pd.DataFrame()
    # mean, standard deviation and variance for each observation across seeds:
    stats["mean"] = df.mean()
    stats["st_dev"] = df.std()
    stats["var"] = df.var()
    print("Creating summary dataset of shape {}".format(stats.shape))
    df.to_csv("raw_seed_datset.csv")
    stats.to_csv("seed_stats.csv")
    prop_high_stdev = sum(stats["st_dev"] > 2) / len(stats["st_dev"])

    logging.info("Proportion of obervations with Standard Deviation > 2: {}".
                 format(prop_high_stdev))

    logging.info("Mean cross-validation score: {}".format(mean_cv_score))


if __name__ == '__main__':
    n_seeds = 30

    make_seed_dataset(XGBRegressor, xgb_params, n_seeds, seed_param="seed")
    logging.info("end of seed test")
