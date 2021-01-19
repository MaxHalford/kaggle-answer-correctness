# kaggle-answer-correctness

This is my solution to the [Riiid! Answer Correctness Prediction competition on Kaggle](https://www.kaggle.com/c/riiid-test-answer-prediction/). I gave waaaaaay before the end so I didn't do very at all on the private leaderboard. However, I'm quite happy with the code quality of this project. It's a solid reference for implementing online feature extractors.

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
