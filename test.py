# import json
# with open('/data/fanli/workspace/Datasets/fitzpatrick17k/meta_data/5_fold_splits.json') as f:
#     data = json.load(f)
# print('可用的折数:', list(data.keys()))


# import json
# with open('/data/fanli/workspace/Datasets/fitzpatrick17k/meta_data/5_fold_splits.json') as f:
#     data = json.load(f)
# print('可用分折:', list(data.keys()))
# print('fold_0验证集样本数:', len(data['fold_0']['valid']))


import pandas as pd
data_df = pd.read_csv("/data/fanli/workspace/Datasets/DDI/meta_data/ddi_metadata.csv")
print(data_df.columns)
print(data_df.head())

