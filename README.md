# kaggle-answer-correctness

Solution to the Riiid! Answer Correctness Prediction competition on Kaggle

## Reading list

- https://paperswithcode.com/task/knowledge-tracing
- https://arxiv.org/pdf/1912.03072.pdf
- https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/188901
- https://www.wikiwand.com/en/Harmonic_mean
- https://arxiv.org/pdf/2002.07033.pdf

## To do

- Check types and missing values (use example test set first)
- Add asserts to check data during test phase
- There are questions in the test set that don't appear in the training set, this has to be handled.
- Consider using the last group of each user in the training set for bootstrapping the test phase
- Make a graph to print the usage periods of a user
- Extract features from a (deep) factorization machine
- Extract features from a graph representation
- Figure out time by reading [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189351#) and [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189465)
- Make sure extractors that should be stateful are in fact stateful
- Learn and use optuna
- Include `type_of` feature, maybe combine it with `part`?

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

kaggle kernels push --path kernel
kaggle kernels status riiid-test-answer-prediction-kernel
open https://www.kaggle.com/maxhalford/riiid-test-answer-prediction-kernel
```

Note that the dataset needs creating for the first upload:

```sh
kaggle datasets init --path dataset
```

Training until validation scores don't improve for 20 rounds
[100]   fit's auc: 0.747558     val's auc: 0.747522
[200]   fit's auc: 0.749077     val's auc: 0.748978
[300]   fit's auc: 0.750209     val's auc: 0.750058
[400]   fit's auc: 0.751028     val's auc: 0.750815
[500]   fit's auc: 0.7516       val's auc: 0.751331
[600]   fit's auc: 0.751998     val's auc: 0.751642
[700]   fit's auc: 0.752271     val's auc: 0.751832
[800]   fit's auc: 0.752479     val's auc: 0.751953
[900]   fit's auc: 0.752637     val's auc: 0.752031
[1000]  fit's auc: 0.752767     val's auc: 0.752081
[1100]  fit's auc: 0.752902     val's auc: 0.752135
[1200]  fit's auc: 0.753033     val's auc: 0.752177
[1300]  fit's auc: 0.753155     val's auc: 0.752215
[1400]  fit's auc: 0.753272     val's auc: 0.752246
[1500]  fit's auc: 0.753387     val's auc: 0.752272
[1600]  fit's auc: 0.7535       val's auc: 0.7523
[1700]  fit's auc: 0.753611     val's auc: 0.752327
[1800]  fit's auc: 0.753724     val's auc: 0.752352
[1900]  fit's auc: 0.75383      val's auc: 0.752376
[2000]  fit's auc: 0.753935     val's auc: 0.752396
[2100]  fit's auc: 0.754038     val's auc: 0.752414
[2200]  fit's auc: 0.754146     val's auc: 0.752433
[2300]  fit's auc: 0.754257     val's auc: 0.752452
[2400]  fit's auc: 0.754365     val's auc: 0.75247
[2500]  fit's auc: 0.754468     val's auc: 0.752482
[2600]  fit's auc: 0.754561     val's auc: 0.752495
[2700]  fit's auc: 0.754662     val's auc: 0.752506
[2800]  fit's auc: 0.754762     val's auc: 0.752519
[2900]  fit's auc: 0.754859     val's auc: 0.75253
[3000]  fit's auc: 0.754962     val's auc: 0.752537
Early stopping, best iteration is:
[3040]  fit's auc: 0.755        val's auc: 0.75254
question_difficulty           16091354
avg_correct                    4318988
user_expo_avg_correct           777277
user_question_count             255582
user_question_avg_duration      176852
timestamp                       155585
part                            148109
user_lecture_count              116211
bundle_size                      66972
bundle_position                   7822
