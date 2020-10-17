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
[100]   fit's auc: 0.746696     val's auc: 0.74666
[200]   fit's auc: 0.747205     val's auc: 0.747097
[300]   fit's auc: 0.7476       val's auc: 0.747462
[400]   fit's auc: 0.74789      val's auc: 0.747739
[500]   fit's auc: 0.748086     val's auc: 0.74792
[600]   fit's auc: 0.748234     val's auc: 0.748035
[700]   fit's auc: 0.748344     val's auc: 0.748102
[800]   fit's auc: 0.748419     val's auc: 0.748132
[900]   fit's auc: 0.748497     val's auc: 0.748171
[1000]  fit's auc: 0.748573     val's auc: 0.748206
[1100]  fit's auc: 0.748647     val's auc: 0.74824
[1200]  fit's auc: 0.748721     val's auc: 0.74827
[1300]  fit's auc: 0.748784     val's auc: 0.748286
[1400]  fit's auc: 0.748831     val's auc: 0.748294
Early stopping, best iteration is:
[1403]  fit's auc: 0.748832     val's auc: 0.748294
