from pathlib import Path

import pandas as pd


data_path = Path("data/raw/enquetes/MDHR81FL.DTA")
df = pd.read_stata(data_path)

print(df.head())
print(df.info())
