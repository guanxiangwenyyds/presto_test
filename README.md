# test presto

## Environment Setup
```bash
git clone https://github.com/guanxiangwenyyds/presto-test.git
cd presto-test
pip install -r requirements.txt
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
feature_extract.py
```
computed feature will be saved in output/temp
