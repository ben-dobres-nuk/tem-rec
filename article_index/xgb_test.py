import pandas as pd
import numpy as np
import random

from xgboost import XGBRegressor

n_seeds = 2

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

all_data = pd.read_pickle('all_data_reindex.pkl')
data_sample = all_data.sample(frac=0.1)

X = data_sample[yaml_feature_names]
y = data_sample['avg_dwell_time_subscriber_read']
print("X shape:")
print(X.shape)
print("      ")

# generate seeds for random sample later
# seed2 = random.sample(range(1, X.shape - 1), 1000)


def trial_seed(n):
    xgb = XGBRegressor(
        max_depth=3,
        seed=n,
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

    xgb.fit(X, y)
    score = xgb.score(X, y)
    text_output = "Score for seed {} is {}".format(n, score)
    print(text_output)

    return (xgb.predict(X))


def make_seed_dataset(n_seeds):
    # sample again to give us a small enough dataset for EDA
    trials = []
    for n in range(1, n_seeds):
        trials.append(trial_seed(n))

    df = pd.DataFrame(
        np.vstack(trials),
        columns=["row" + str(x) for x in range(1, y.shape[0] + 1)])
    print("Creating seed dataset of shape {}".format(df.shape))
    stats = pd.DataFrame()
    stats["mean"] = df.mean(axis=0)
    stats["Std.Dev"] = df.std(axis=0)
    stats["Var"] = df.var(axis=0)
    df.to_csv("raw_seed_datset.csv")
    stats.to_csv("seed_stats.csv")


if __name__ == '__main__':
    make_seed_dataset(n_seeds)
