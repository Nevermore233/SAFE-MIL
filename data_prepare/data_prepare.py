import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import random
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# load data
data_clinical = pd.read_csv("input/data_clinical_input.csv")
data_tissue = pd.read_csv("input/data_tissue_first_line_treatment_input.csv")

# add features
# TP53
tp53_tissue_ids = set(data_tissue[data_tissue['gene'] == 'TP53']['tissue_id'])
data_clinical['TP53'] = 0
for index, row in data_clinical.iterrows():
    if row['tissue_id'] in tp53_tissue_ids:
        data_clinical.at[index, 'TP53'] = 1

# number of co-mutated genes
tissue_counts = data_tissue['tissue_id'].value_counts().to_dict()
data_clinical['number of co-mutated genes'] = data_clinical['tissue_id'].map(tissue_counts)

# data_clinical preprocessing
columns = ['patient_id', 'tissue_id', 'age', 'gender', 'stage', 'smoking', 'medication',
           'pfs', 'best therapeutic tesponse', "progresses", "TP53", "number of co-mutated genes"]
data_clinical = data_clinical[data_clinical.columns.intersection(columns)]
data_clinical['progresses'] = data_clinical['progresses'].replace({'yes': 1, 'no': 0})
data_clinical['best therapeutic tesponse'] = data_clinical['best therapeutic tesponse'].replace(
    {'PR': 0, 'SD': 1, 'PD': 2})
data_clinical['medication'], _ = pd.factorize(data_clinical['medication'])
data_clinical['gender'] = data_clinical['gender'].replace({'F': 1, 'M': 0})
data_clinical['stage'], _ = pd.factorize(data_clinical['stage'])
data_clinical['smoking'], _ = pd.factorize(data_clinical['smoking'])
data_clinical = data_clinical.drop_duplicates(subset='patient_id', keep='first')
data_clinical.reset_index(drop=True)

#  data_tissue preprocessing
columns_ = ['tissue_id', 'gene', 'mutation abundance', "caseDep", 'caseAltDep']
data_tissue = data_tissue[data_tissue.columns.intersection(columns_)]
data_tissue = data_tissue.dropna(subset=['tissue_id'])
data_tissue = data_tissue[data_tissue['gene'] == 'EGFR']
data_tissue = data_tissue.drop('gene', axis=1)
data_tissue['mutation abundance'] = data_tissue['mutation abundance'].str.replace('%', '')
data_tissue['mutation abundance'] = data_tissue['mutation abundance'].astype(float)
data_tissue = data_tissue.drop_duplicates(subset='tissue_id', keep='first')
data_tissue = data_tissue.astype(float)

# merge
common_tissue_ids = data_clinical[data_clinical['tissue_id'].isin(data_tissue['tissue_id'])]['tissue_id']
data_clinical = data_clinical[data_clinical['tissue_id'].isin(common_tissue_ids)]
print(data_clinical.shape)
data_tissue = data_tissue[data_tissue['tissue_id'].isin(common_tissue_ids)]
print(data_tissue.shape)
df_merged = pd.merge(data_clinical, data_tissue, on='tissue_id', how='inner')

#  Generate label
df_merged['label'] = float('nan')
scaler = MinMaxScaler()
pfs_values = df_merged['pfs'].values.reshape(-1, 1)
normalized_pfs = scaler.fit_transform(pfs_values)
df_merged['label'] = normalized_pfs

data = df_merged[['age', 'gender', 'mutation abundance', 'TP53', 'number of co-mutated genes', 'pfs', 'label']]
cols_to_normalize = ['age', 'mutation abundance']
min_values = data[cols_to_normalize].min()
max_values = data[cols_to_normalize].max()
print('\nmax_values:\n', max_values)
print('\nmin_values:\n', min_values)
df_normalized = (data[cols_to_normalize] - min_values) / (max_values - min_values)
data[cols_to_normalize] = df_normalized

features = data[['age', 'gender', 'mutation abundance', 'TP53', 'number of co-mutated genes', 'label']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_features)
cluster_labels = kmeans.labels_
data['cluster_label'] = cluster_labels

# Generate 600 bags
MIL_600 = pd.DataFrame()
bag_num = 1
for j in range(0, 4):
    cluster_data = data[data['cluster_label'] == j]

    for i in range(1, 151):
        bag_name = 'bag' + str(bag_num)

        bag_samples = random.randint(1, 10)
        samples = cluster_data.sample(n=bag_samples, replace=True)
        bag_ma = samples['mutation abundance'].mean()
        bag_label = samples['label'].mean()

        bag_data = pd.DataFrame(samples.values, columns=samples.columns)
        bag_data['bag_ma'] = bag_ma
        bag_data['bag_names'] = bag_name
        bag_data['bag_labels'] = bag_label

        MIL_600 = pd.concat([MIL_600, bag_data], ignore_index=True)
        bag_num += 1

pfs_avg = MIL_600.groupby('bag_names')['pfs'].mean().reset_index()
ma_avg = MIL_600.groupby('bag_names')['bag_ma'].mean().reset_index()

MIL_600 = MIL_600[['gender', 'age', 'mutation abundance', 'TP53', 'number of co-mutated genes', 'label', 'bag_names', 'bag_labels']]
MIL_600.reset_index(drop=True, inplace=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file1 = f"output/MIL_600_{current_time}.csv"
MIL_600.to_csv(output_file1)

output_file2 = f"output/pfs_avg_{current_time}.csv"
pfs_avg.to_csv(output_file2)

output_file3 = f"output/ma_avg_{current_time}.csv"
ma_avg.to_csv(output_file3)
print('Generate 600 bags finished!')

# Generate 1000 bags
MIL_1000 = pd.DataFrame()
bag_num = 1
for j in range(0, 4):
    cluster_data = data[data['cluster_label'] == j]

    for i in range(1, 251):
        bag_name = 'bag' + str(bag_num)
        bag_samples = random.randint(1, 10)
        samples = cluster_data.sample(n=bag_samples, replace=True)
        bag_ma = samples['mutation abundance'].mean()
        bag_label = samples['label'].mean()

        bag_data = pd.DataFrame(samples.values, columns=samples.columns)
        bag_data['bag_ma'] = bag_ma
        bag_data['bag_names'] = bag_name
        bag_data['bag_labels'] = bag_label

        MIL_1000 = pd.concat([MIL_1000, bag_data], ignore_index=True)
        bag_num += 1

MIL_1000 = MIL_1000[['gender', 'age', 'mutation abundance', 'TP53', 'number of co-mutated genes', 'label', 'bag_names', 'bag_labels']]
MIL_1000.reset_index(drop=True, inplace=True)

output_file4 = f"output/MIL_1000_{current_time}.csv"
MIL_1000.to_csv(output_file4)
print('Generate 1000 bags finished!')







