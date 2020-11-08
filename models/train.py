import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection


nrows = None  # set to a number for debugging
features = pd.concat(
    (
        pd.read_csv(f, index_col=0, nrows=nrows)
        for f in glob.glob('features/*.csv')
    ),
    axis='columns'
)
features['part'] = features['part'].astype('category')
print(features.head())

# Let's join the target variable.
targets = pd.read_csv(
    'data/train.csv',
    usecols=['row_id', 'answered_correctly'],
    index_col='row_id',
    nrows=nrows
)
features = features.join(targets)
targets = features.pop('answered_correctly')
features.head()

np.random.seed(42)
samples = np.random.choice(targets.index, size=3_000_000, replace=False)
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(
    features.loc[samples], targets.loc[samples],
    random_state=42
)
# X_fit, X_val, y_fit, y_val = model_selection.train_test_split(
#    features, targets,
#    random_state=42
# )
fit = lgb.Dataset(X_fit, y_fit)
val = lgb.Dataset(X_val, y_val, reference=fit)

model = lgb.train(
    params={
        'learning_rate': 0.05,
        'objective': 'binary',
        'metric': 'auc',
        'boost_from_average': True,
        'max_bin': 800,
        'num_leaves': 80
    },
    train_set=fit,
    num_boost_round=10_000,
    valid_sets=(fit, val),
    valid_names=('fit', 'val'),
    early_stopping_rounds=20,
    verbose_eval=100
)

importances = model.feature_importance(importance_type='gain').astype(int)
print(pd.Series(importances, index=X_fit.columns).sort_values(ascending=False))

val_score = metrics.roc_auc_score(y_val, model.predict(X_val))
model.save_model(f'models/model_{val_score:.4f}.lgb')
