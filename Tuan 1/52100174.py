import pandas as pd
df=pd.read_csv('iris.data')
#  Data cleaning
df=df.dropna()
df=df.drop_duplicates()
df=df.iloc[: , :-1]

#  Data normalization

for column in df.columns:
  min_value = df[column].min()
  max_value = df[column].max()
  df[column] = (df[column] - min_value) / (max_value - min_value)

# Data transformation 

df.rename(columns={'5.1':'col-1','3.5':'col-2','1.4':'col-3','0.2':'col-4'},inplace=True)
print(df.to_string())
