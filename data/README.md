# Datasets

Place the MATH dataset (Hendrycks et al., 2021) here.

## Download

```bash
# Clone the official MATH repo
git clone https://github.com/hendrycks/math.git /tmp/math_repo

# Convert to HuggingFace datasets format and save here
python -c "
from datasets import Dataset, DatasetDict
import json, os, glob

data_root = '/tmp/math_repo/MATH'
for split in ['train', 'test']:
    for subject in os.listdir(os.path.join(data_root, split)):
        subj_path = os.path.join(data_root, split, subject)
        if not os.path.isdir(subj_path): continue
        examples = []
        for fname in glob.glob(os.path.join(subj_path, '*.json')):
            with open(fname) as f:
                ex = json.load(f)
                examples.append({
                    'problem': ex['problem'],
                    'solution': ex['solution'],
                    'level': ex['level'],
                    'type': ex.get('type', ''),
                })
        ds = Dataset.from_list(examples)
        ds.save_to_disk(f'data/math/{subject}/{split}')
"
```

## Structure

After download, the directory should look like:

```
data/math/
├── algebra/
│   ├── train/
│   └── test/
├── geometry/
│   ├── train/
│   └── test/
├── ...
└── precalculus/
    ├── train/
    └── test/
```
