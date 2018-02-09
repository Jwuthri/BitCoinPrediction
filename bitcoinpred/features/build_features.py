# -*- coding: utf-8 -*-
"""
@author: JulienWuthrich
"""
import os

import pandas as pd

from bitcoinpred.config.settings import raw_data_path, converged_data_path


path1 = os.path.join(raw_data_path, "sentiment6.txt")
path2 = os.path.join(raw_data_path, "bitcoinprices.txt")
df1 = pd.read_csv(path1)
df1.columns = ["stamp", "price"]
df2 = pd.read_csv(path2)
df2.columns = ["stamp", "sentiment"]
df = pd.merge(df1, df2, on="stamp", how="inner")
path = os.path.join(converged_data_path, "merged.csv")
df.to_csv(path, index=False)
