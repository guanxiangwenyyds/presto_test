# test presto

The code includes all the processes from the raw data processing to the final experiment for "Feature_Extraction_Testing_Based_on_the_Presto_Model", the dataset used is already present in the package, follow the steps below to run the whole process step by step.

## Environment Setup
```bash
git clone https://github.com/guanxiangwenyyds/presto-test.git
cd presto-test
conda env create -f environment.yml
conda activate presto_test
```
In the downloaded package, the test_data/BigEarthNet directory already contains the original dataset used in this experiment, which is a subset of the entire BigEarthNet-S2 dataset and contains satellite data for single time point. The whole BigEarthNet-S2 dataset can be downloaded from https://bigearth.net/

## Data preparation
The code already contains the processed dataset as well as the labels and this step can be skipped.

If you still want to run the code for dataset processing:
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
