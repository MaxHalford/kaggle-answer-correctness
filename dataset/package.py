import glob
import json
import os
import pathlib
import shutil

# Remove everything
for file in pathlib.Path('dataset').iterdir():
    if file.name in ('.gitkeep', 'package.py'):
        continue
    file.unlink()

# Feature extractors
for extractor in glob.glob('features/*.pkl'):
    shutil.copy(extractor, 'dataset')
shutil.copy('features/module.py', 'dataset')

# Model
models = glob.glob('models/*.lgb')
latest_model = max(models, key=os.path.getctime)
shutil.copy(latest_model, 'dataset')

# Metadata
with open('dataset/dataset-metadata.json', 'w') as f:
    meta = {
        'title': 'riiid-test-answer-prediction-dataset',
        'id': 'maxhalford/riiid-test-answer-prediction-dataset',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    json.dump(meta, f, indent=4)
