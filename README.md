# kaggle-answer-correctness

Solution to the Riiid! Answer Correctness Prediction competition on Kaggle

## Reading list

- https://paperswithcode.com/task/knowledge-tracing
- https://arxiv.org/pdf/1912.03072.pdf
- https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/188901
- https://www.wikiwand.com/en/Harmonic_mean

## To do

- Check types and missing values (use example test set first)
- Add asserts to check data during test phase
- There are questions in the test set that don't appear in the training set, this has to be handled.
- Consider using the last group of each user in the training set for bootstrapping the test phase
- Make a graph to print the usage periods of a user
- Extract features from a (deep) factorization machine
- Extract features from a graph representation
- Figure out time by reading [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189351#) and [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189465)
- Exponentially weighted means

## Reproducing results

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt

rm features/*.csv

python features/extract.py
python models/train.py

python dataset/package.py
kaggle datasets version --path dataset --message "$(date)" --delete-old-versions

python kernel/package.py
kaggle kernels push --path kernel
kaggle kernels status riiid-test-answer-prediction-kernel
open https://www.kaggle.com/maxhalford/riiid-test-answer-prediction-kernel
```

Note that the dataset needs creating for the first upload:

```sh
kaggle datasets init --path dataset
```

Training until validation scores don't improve for 20 rounds
[100]   fit's auc: 0.746732     val's auc: 0.746691
[200]   fit's auc: 0.747706     val's auc: 0.747595
[300]   fit's auc: 0.748557     val's auc: 0.748429
[400]   fit's auc: 0.749175     val's auc: 0.749036
[500]   fit's auc: 0.749642     val's auc: 0.749472
[600]   fit's auc: 0.749978     val's auc: 0.749756
[700]   fit's auc: 0.750197     val's auc: 0.749918
[800]   fit's auc: 0.750352     val's auc: 0.750007
[900]   fit's auc: 0.750475     val's auc: 0.750067
[1000]  fit's auc: 0.750587     val's auc: 0.750116
[1100]  fit's auc: 0.750684     val's auc: 0.750155
[1200]  fit's auc: 0.750769     val's auc: 0.750183
[1300]  fit's auc: 0.750857     val's auc: 0.750212
[1400]  fit's auc: 0.75094      val's auc: 0.750237
[1500]  fit's auc: 0.751024     val's auc: 0.750257
[1600]  fit's auc: 0.751106     val's auc: 0.750277
[1700]  fit's auc: 0.751187     val's auc: 0.750294
[1800]  fit's auc: 0.751266     val's auc: 0.750309
[1900]  fit's auc: 0.751348     val's auc: 0.750323
[2000]  fit's auc: 0.751429     val's auc: 0.750339
[2100]  fit's auc: 0.751505     val's auc: 0.750353
[2200]  fit's auc: 0.751582     val's auc: 0.750365
[2300]  fit's auc: 0.75166      val's auc: 0.750377
[2400]  fit's auc: 0.751732     val's auc: 0.750385
[2500]  fit's auc: 0.751806     val's auc: 0.750393
[2600]  fit's auc: 0.751878     val's auc: 0.750399
Early stopping, best iteration is:
[2651]  fit's auc: 0.751914     val's auc: 0.750402
