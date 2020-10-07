import glob

import chime
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
print(features.head())
chime.info()

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

chime.info()
model = lgb.train(
    params={
        'learning_rate': 0.01,
        'objective': 'binary',
        'metrics': 'auc, logloss',
        'boost_from_average': False
    },
    train_set=fit,
    num_boost_round=10_000,
    valid_sets=(fit, val),
    valid_names=('fit', 'val'),
    early_stopping_rounds=20,
    verbose_eval=100
)

val_score = metrics.roc_auc_score(y_val, model.predict(X_val))
model.save_model(f'models/model_{val_score:.4f}.lgb')
chime.success()
