import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join("data/hat-ios", "Report1550_e.csv")

rawData = pd.read_csv(DATA_PATH)

rawData.info()

# rawData.dropna(subset=['SHOW'])
# rawData.describe()
# rawData.hist(bins=50, figsize=(20, 15))
# plt.show()

# for i in range(len(rawData)):
#     if rawData.ix[i, 'evar96'] == 'SHOW':
#         rawData = rawData.drop(i)
#
# rawData.info()

# checkNotShowData = rawData.copy()
#
# checkNotShowData.dropna(subset=["SHOW"])
# checkNotShowData.info()
# checkNotShowData.drop().where(checkNotShowData["evar96"] = "")

