import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join("data", "Report1550_e.csv")

rawData = pd.DataFrame.from_csv(DATA_PATH)

rawData.info()

rawData.dropna(subset=['SHOW'])
rawData.info()
# rawData.hist(bins=50, figsize=(20, 15))
# plt.show()

# checkNotShowData = rawData.copy()
#
# checkNotShowData.dropna(subset=["SHOW"])
# checkNotShowData.info()
# checkNotShowData.drop().where(checkNotShowData["evar96"] = "")

