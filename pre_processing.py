import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the dataset
list_of_csv_files = ['Monday.csv', 'processedFriday.csv', 'processedFridaymorn.csv', 'processedFridayport.csv']
path = 'data/'
dataset = pd.concat([pd.read_csv(path+file) for file in list_of_csv_files])

# Handle missing values
imputer = SimpleImputer(strategy="mean")
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# Drop too large values
max_threshold = dataset.quantile(0.95)
dataset = dataset[dataset < max_threshold]

# Drop too small values
min_threshold = dataset.quantile(0.05)
dataset = dataset[dataset > min_threshold]

# Dealing with outliers
from scipy import stats
z_scores = stats.zscore(dataset)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
dataset = dataset[filtered_entries]

# Min-max normalization
scaler = MinMaxScaler()
dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

# One-hot encoding for labels
dataset = pd.get_dummies(dataset, columns=['Label'])

# Save preprocessed dataset
dataset.to_csv("normalised/Concatenated_dataset.csv", index=False)
