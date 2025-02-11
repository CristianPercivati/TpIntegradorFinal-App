import pandas as pd

df = pd.read_csv('serving\\outputs\\asd2_output.csv')

print(df.value_counts('Ubicación'))