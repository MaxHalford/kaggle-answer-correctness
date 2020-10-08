import json
import shutil

# Notebook
shutil.copy('Testing phase.ipynb', 'kernel')

# Metadata
with open('kernel/kernel-metadata.json', 'w') as f:
    meta = {
        'id': 'maxhalford/riiid-test-answer-prediction-kernel',
        'title': 'riiid-test-answer-prediction-kernel',
        'code_file': 'Testing phase.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': 'false',
        'enable_gpu': 'false',
        'enable_internet': 'false',
        'dataset_sources': ['maxhalford/riiid-test-answer-prediction-dataset'],
        'competition_sources': ['riiid-test-answer-prediction'],
        'kernel_sources': []
    }
    json.dump(meta, f, indent=4)
