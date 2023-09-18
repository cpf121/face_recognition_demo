import pandas as pd

print(pd.Series([2, 4, 6, 8, 10]))

s1 = pd.Series([2, 4, 6, 8])
print(s1[0])

#  dataFrame
df = pd.DataFrame({
    '辣条': [14, 20],
    '面包': [7, 3],
    '可乐': [8, 13],
    '烤肠': [10, 6]
})

print(df['可乐'])

pd.read_csv('./metadata.csv', sep="|", header=None, quoting=3)
