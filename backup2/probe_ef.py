import pandas as pd

path = './extracted_features/extracted_features_29.csv'
theory_path = '../theory/theory_A_N.csv'
data = pd.read_csv(path)
theory = pd.read_csv(theory_path)

columns = [f'Prob_Conj_{i+1}' for i in range(62)]
columns.extend(['max_conj_#', 'labels', 'prediction', 'kardio_label'])
data.columns = columns

print(data.head())

data = data[['max_conj_#', 'prediction', 'kardio_label']]
print(data['max_conj_#'].value_counts())

# for index, row in theory.iterrows():
#     if index == 58:
#         print(row)