# kaggle-answer-correctness

Solution to the Riiid! Answer Correctness Prediction competition on Kaggle

## Reading list

- https://paperswithcode.com/task/knowledge-tracing
- https://arxiv.org/pdf/1912.03072.pdf
- https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/188901
- https://www.wikiwand.com/en/Harmonic_mean
- [Towards an Appropriate Query, Key, and ValueComputation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf)

## To do

- Consider using the last group of each user in the training set for bootstrapping the test phase
- Make a graph to print the usage periods of a user
- Extract features from a (deep) factorization machine
- Extract features from a graph representation
- Figure out time by reading [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189351#) and [this](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189465)
- Make sure extractors that should be stateful are in fact stateful
- Learn and use optuna
- Include `type_of` feature, maybe combine it with `part`?
- There might be some hard rules for predicting (trivial onboarding questions for instance)
- Check https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/192839
- Check https://www.kaggle.com/dwit392/lgbm-iii
- lag time, the time interval between adjacent learning activities
- elapsed time, the time taken for a student to answer
- At what do users look at corrections when they're wrong

## Reproducing results

```sh
conda create -n kaggle python=3.7.6 -y
conda activate kaggle
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

                                                    update       transform
QuestionDifficulty                                 0:00:00  0:00:08.471614
Part                                               0:00:00  0:00:01.369010
BundleSize                                         0:00:00  0:00:44.378854
BundlePosition                                     0:00:00  0:00:25.248702
Timestamp                                          0:00:00  0:00:00.930190
QuestionNChoices                                   0:00:00  0:00:07.146908
QuestionAnswerEntropy                              0:00:00  0:00:05.968693
UserLectureCount                            0:01:51.114094  0:00:12.234477
UserQuestionCount                           0:02:18.322431  0:00:11.447532
UserExpAvgCorrect_prior_mean=0.5_alpha=0.2  0:02:24.857929  0:00:11.307162
UserPartCount                               0:03:44.622387  0:07:16.949918
AvgCorrect_prior_mean=0.6_prior_size=20     0:03:52.157035  0:00:13.083721
UserQuestionAvgDuration                     0:04:36.966671  0:00:11.863548
DejaVu                                      0:04:38.258979  0:14:25.225591
