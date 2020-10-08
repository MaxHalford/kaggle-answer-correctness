# kaggle-answer-correctness

Solution to the Riiid! Answer Correctness Prediction competition on Kaggle

## Reading list

- https://paperswithcode.com/task/knowledge-tracing
- https://arxiv.org/pdf/1912.03072.pdf
- https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/188901

## To do

- There are questions in the test set that don't appear in the training set, this has to be handled.
- Consider using the last group of each user in the training set for bootstrapping the test phase
- Make a graph to print the usage periods of a user
- Extract features from a (deep) factorization machine

## Reproducing results

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt

rm features/*.csv
python features/extract.py
python models/train.py
python dataset/package.py
python kernel/package.py

kaggle datasets version --path dataset --message "$(date)" --delete-old-versions
kaggle kernels push --path kernel
kaggle kernels status riiid-test-answer-prediction-kernel
open https://www.kaggle.com/maxhalford/riiid-test-answer-prediction-kernel
```

Note that the dataset needs creating for the first upload:

```sh
kaggle datasets init --path dataset
```
