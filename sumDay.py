import pandas as pd
DAWmat=[['Day1', 'arm', 3], ['Day1', 'arm', 2], ['Day1','leg',5],['Day1','arm',6],['Day2', 'leg', 1], ['Day2', 'arm', 4]]
df = pd.DataFrame(DAWmat, columns=['Col1','Col2','Col3'])
gro = df.groupby(['Col1','Col2'], as_index=False)['Col3'].sum()
lol = gro.values.tolist()
lol