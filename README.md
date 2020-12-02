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
- At what rate do users look at corrections when they're wrong

## Reproducing results

```sh
# Use the same environment as on Kaggle kernels
conda create -n kaggle python=3.7.6 -y
conda activate kaggle
pip install -r requirements.txt

# Extract features on the training set and save feature extractors for the test phase
rm features/*.csv features/*.pkl
python features/extract.py

# Train a model and save it
python models/build_training_set.py
python models/train.py

# Compile the files to upload to Kaggle
python dataset/package.py
kaggle datasets version --path dataset --message "$(date)" --delete-old-versions

# Push the kernel associated to the above dataset for the test phase
kaggle kernels push --path kernel
kaggle kernels status riiid-test-answer-prediction-kernel
open https://www.kaggle.com/maxhalford/riiid-test-answer-prediction-kernel
```

Note that the dataset needs creating for the first upload:

```sh
kaggle datasets init --path dataset
```

## Current best

### Timings

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
QuestionCount                                      0:00:00  0:00:06.634506
UserRelevantLectureCount                    0:00:13.460764  0:03:56.869746
UserHadExplanationTotal                     0:02:15.969187  0:00:12.249619
UserQuestionTagCounter                      0:09:01.066594  0:11:33.838792

### Current accuracy

Training until validation scores don't improve for 20 rounds
[100]   fit's auc: 0.763363     val's auc: 0.761949
[200]   fit's auc: 0.766936     val's auc: 0.763994
[300]   fit's auc: 0.768864     val's auc: 0.764326
[400]   fit's auc: 0.770645     val's auc: 0.764575
[500]   fit's auc: 0.772296     val's auc: 0.764721
[600]   fit's auc: 0.773896     val's auc: 0.764828
[700]   fit's auc: 0.775403     val's auc: 0.764879
Early stopping, best iteration is:
[705]   fit's auc: 0.775469     val's auc: 0.764886

### Best local accuracy on 3M observations

0.769 on public LB

[LightGBM] [Info] Start training from score 0.652179
Training until validation scores don't improve for 20 rounds
[100]   fit's auc: 0.762727     val's auc: 0.761305
[200]   fit's auc: 0.765963     val's auc: 0.763118
[300]   fit's auc: 0.767801     val's auc: 0.763441
[400]   fit's auc: 0.769393     val's auc: 0.763601
[500]   fit's auc: 0.770966     val's auc: 0.763693
Early stopping, best iteration is:
[513]   fit's auc: 0.77116      val's auc: 0.763706

user_question_avg_correct_harmonic    4023011
deja_vu_incorrect                      187768
question_avg_correct                   171750
user_expo_avg_correct                  101648
user_avg_correct                       100944
part_incorrect                          85125
deja_vu_correct                         83220
user_question_count                     79204
part_correct                            65508
timestamp                               54553
question_answer_entropy                 51998
part                                    48007
user_question_avg_duration              41308
user_lecture_count                      37969
bundle_size                              8136
bundle_position                          2164
dtype: int64
