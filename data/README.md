# Datasets

## SMAP & MSL (NASA)

Download from the official Telemanom repository:

```bash
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip -d ./data
```

Expected structure after extraction:
```
data/
  train/   # .npy files per entity
  test/    # .npy files per entity
  labeled_anomalies.csv
```

Then preprocess:
```bash
python scripts/preprocess.py --dataset smap --data_dir ./data
python scripts/preprocess.py --dataset msl  --data_dir ./data
```

## ESA-AD

The ESA anomaly detection benchmark used in the original PSTG paper.
See: https://github.com/ESA-PhiLab/ESA-ADB
