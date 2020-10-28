# kaggle-answer-correctness

Solution to the Riiid! Answer Correctness Prediction competition on Kaggle

## Reading list

- https://paperswithcode.com/task/knowledge-tracing
- https://arxiv.org/pdf/1912.03072.pdf
- https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/188901
- https://www.wikiwand.com/en/Harmonic_mean
- [Towards an Appropriate Query, Key, and ValueComputation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf)

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
- There might be some hard rules for predicting (trivial onboarding questions for instance)
- Check https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/192839
- Check https://www.kaggle.com/dwit392/lgbm-iii
- lag time, the time interval between adjacent learning activities
- elapsed time, the time taken for a student to answer
- Add entropy:

    question_answer_dist = train.query('content_type_id == 0').groupby(['content_id', 'user_answer']).size()
    question_answer_entropy = question_answer_dist.groupby('content_id').apply(lambda counts: stats.entropy(counts + 50))
    question_answer_entropy

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




    [100]   fit's auc: 0.752639     val's auc: 0.752683
    [200]   fit's auc: 0.755511     val's auc: 0.755396
    [300]   fit's auc: 0.757398     val's auc: 0.757196
    [400]   fit's auc: 0.758675     val's auc: 0.758393
    [500]   fit's auc: 0.759565     val's auc: 0.759208
    [600]   fit's auc: 0.760111     val's auc: 0.759681
    [700]   fit's auc: 0.760476     val's auc: 0.759971
    [800]   fit's auc: 0.760727     val's auc: 0.760152
    [900]   fit's auc: 0.760922     val's auc: 0.760267
    [1000]  fit's auc: 0.761075     val's auc: 0.760343
    [1100]  fit's auc: 0.761214     val's auc: 0.760403
    [1200]  fit's auc: 0.761347     val's auc: 0.760454
    [1300]  fit's auc: 0.761472     val's auc: 0.760497
    [1400]  fit's auc: 0.761598     val's auc: 0.760542
    [1500]  fit's auc: 0.761716     val's auc: 0.760582
    [1600]  fit's auc: 0.761841     val's auc: 0.760622
    [1700]  fit's auc: 0.761964     val's auc: 0.760658
    [1800]  fit's auc: 0.762084     val's auc: 0.760687
    [1900]  fit's auc: 0.762207     val's auc: 0.760719
    [2000]  fit's auc: 0.76232      val's auc: 0.760745
    [2100]  fit's auc: 0.76243      val's auc: 0.760769
    [2200]  fit's auc: 0.762537     val's auc: 0.76079
    [2300]  fit's auc: 0.762646     val's auc: 0.760811
    [2400]  fit's auc: 0.762754     val's auc: 0.760832
    [2500]  fit's auc: 0.762859     val's auc: 0.760847
    [2600]  fit's auc: 0.762963     val's auc: 0.760859
    [2700]  fit's auc: 0.763057     val's auc: 0.760868
    [2800]  fit's auc: 0.763149     val's auc: 0.760878
    [2900]  fit's auc: 0.763245     val's auc: 0.760889
    [3000]  fit's auc: 0.76334      val's auc: 0.760899
    [3100]  fit's auc: 0.763432     val's auc: 0.760906
    [3200]  fit's auc: 0.763522     val's auc: 0.760915
    [3300]  fit's auc: 0.763617     val's auc: 0.760924
    [3400]  fit's auc: 0.763712     val's auc: 0.760931
    [3500]  fit's auc: 0.763809     val's auc: 0.760944
    [3600]  fit's auc: 0.763907     val's auc: 0.760956
    [3700]  fit's auc: 0.764004     val's auc: 0.760968
    [3800]  fit's auc: 0.764098     val's auc: 0.760981
    [3900]  fit's auc: 0.764187     val's auc: 0.760986
    [4000]  fit's auc: 0.764278     val's auc: 0.760993
    [4100]  fit's auc: 0.764363     val's auc: 0.760999
    Early stopping, best iteration is:
    [4173]  fit's auc: 0.764427     val's auc: 0.761005
    question_difficulty           16208101
    avg_correct                    4250706
    deja_vu_incorrect               990704
    user_expo_avg_correct           793541
    deja_vu_correct                 432402
    user_question_count             359160
    timestamp                       217837
    part                            210127
    user_question_avg_duration      200103
    user_lecture_count              158292
    bundle_size                      54968
    bundle_position                  10146
    question_n_choices                  32


[100]   fit's auc: 0.752652     val's auc: 0.752693
[200]   fit's auc: 0.755519     val's auc: 0.755401
[300]   fit's auc: 0.757406     val's auc: 0.757212
[400]   fit's auc: 0.7587       val's auc: 0.758427
[500]   fit's auc: 0.759609     val's auc: 0.759264
[600]   fit's auc: 0.760181     val's auc: 0.759769
[700]   fit's auc: 0.760565     val's auc: 0.760082
[800]   fit's auc: 0.760836     val's auc: 0.760275
[900]   fit's auc: 0.761032     val's auc: 0.760393
[1000]  fit's auc: 0.761206     val's auc: 0.760484
[1100]  fit's auc: 0.761357     val's auc: 0.760556
[1200]  fit's auc: 0.761488     val's auc: 0.760606
[1300]  fit's auc: 0.76162      val's auc: 0.760655
[1400]  fit's auc: 0.761754     val's auc: 0.760704
[1500]  fit's auc: 0.761881     val's auc: 0.760749
[1600]  fit's auc: 0.762        val's auc: 0.760783
[1700]  fit's auc: 0.762126     val's auc: 0.760822
[1800]  fit's auc: 0.76225      val's auc: 0.760858
[1900]  fit's auc: 0.762372     val's auc: 0.760892
[2000]  fit's auc: 0.762497     val's auc: 0.760927
[2100]  fit's auc: 0.762616     val's auc: 0.760955
[2200]  fit's auc: 0.76273      val's auc: 0.760985
[2300]  fit's auc: 0.762843     val's auc: 0.76101
[2400]  fit's auc: 0.76296      val's auc: 0.761038
[2500]  fit's auc: 0.763077     val's auc: 0.761064
[2600]  fit's auc: 0.763187     val's auc: 0.761081
[2700]  fit's auc: 0.76329      val's auc: 0.761097
[2800]  fit's auc: 0.763397     val's auc: 0.761118
[2900]  fit's auc: 0.763503     val's auc: 0.761132
[3000]  fit's auc: 0.763608     val's auc: 0.761144
[3100]  fit's auc: 0.763715     val's auc: 0.761155
[3200]  fit's auc: 0.76382      val's auc: 0.761167
[3300]  fit's auc: 0.763921     val's auc: 0.761181
[3400]  fit's auc: 0.764018     val's auc: 0.761195
[3500]  fit's auc: 0.764121     val's auc: 0.761208
[3600]  fit's auc: 0.76422      val's auc: 0.761221
[3700]  fit's auc: 0.764317     val's auc: 0.761236
[3800]  fit's auc: 0.764414     val's auc: 0.761248
[3900]  fit's auc: 0.764503     val's auc: 0.761257
[4000]  fit's auc: 0.764601     val's auc: 0.761266
[4100]  fit's auc: 0.764694     val's auc: 0.761277
[4200]  fit's auc: 0.764789     val's auc: 0.761285
[4300]  fit's auc: 0.764888     val's auc: 0.761296
[4400]  fit's auc: 0.764986     val's auc: 0.761304
[4500]  fit's auc: 0.765078     val's auc: 0.761309
[4600]  fit's auc: 0.765177     val's auc: 0.76132
[4700]  fit's auc: 0.765271     val's auc: 0.761328
[4800]  fit's auc: 0.765374     val's auc: 0.761335
[4900]  fit's auc: 0.765474     val's auc: 0.761341
[5000]  fit's auc: 0.765568     val's auc: 0.761346
[5100]  fit's auc: 0.765663     val's auc: 0.761354
[5200]  fit's auc: 0.76576      val's auc: 0.76136
[5300]  fit's auc: 0.765865     val's auc: 0.761375
[5400]  fit's auc: 0.765954     val's auc: 0.761385
[5500]  fit's auc: 0.766042     val's auc: 0.76139
Early stopping, best iteration is:
[5493]  fit's auc: 0.766036     val's auc: 0.761391
