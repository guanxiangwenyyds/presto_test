# test presto

## Environment Setup
```bash
git clone https://github.com/guanxiangwenyyds/presto-test.git
cd presto-test
conda env create -f environment.yml
```

## Data preparation
The code already contains the processed dataset as well as the labels and this step can be skipped.
If you want to run the code for dataset processing:
```bash
python dataset_process.py
```

## Feature extraction
After runing dataset_process.py, or directly do feature extraction:
```bash
python feature_extract.py
```
computed feature will be saved in presto-test/output/temp

## Experiment
After successfully running feature_extract.py, can start a series of tests.

### UMAP
```bash
python UMAP.py
```

### Linear Probing
```bash
python linear_probing.py
```

### Classification
```bash
python classification.py
```
