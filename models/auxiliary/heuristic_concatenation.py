import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(codebook_path, heart_failure_path):
    codebook = pd.read_csv(codebook_path)
    heart_failure = pd.read_csv(heart_failure_path)
    #heart_failure['sex'] = heart_failure['sex'].map({1: 'M', 0: 'F'})

    return codebook, heart_failure

def bin_based_concatentaion(codebook, heart_failure):

    bins = [i for i in range(0, 101, 10)]  # 0-10, 10-20, 20-30, ..., 90-100
    labels = [f'{i}-{i+10}' for i in range(0, 100, 10)]  # Label for each bin


    heart_failure['age_bin'] = pd.cut(heart_failure['age'], bins=bins, labels=labels, right=False)

    mean_values = heart_failure.groupby('age_bin', observed=False).mean().reset_index()

    mean_values_array = mean_values.drop('age_bin', axis=1)


    column_names = mean_values.drop('age_bin', axis=1).columns


    mean_values_heart_failure = pd.DataFrame(mean_values_array.values, columns=column_names)

    mean_values_heart_failure['age_bin'] = mean_values['age_bin'].values

    def find_bin_for_age(age):
        for idx, row in mean_values_heart_failure.iterrows():
            age_range = row['age_bin']
            start_age, end_age = map(int, age_range.split('-'))
            if start_age <= age < end_age:
                return row[['anaemia','creatinine_phosphokinase','diabetes','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','DEATH_EVENT']] 
        return None

    codebook[['anaemia','creatinine_phosphokinase','diabetes','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','DEATH_EVENT']] = codebook['Age'].apply(lambda age: find_bin_for_age(age)).apply(pd.Series)

    return codebook

def main():
    codebook_path = 'codebook.csv' # Path to codebook.csv
    heart_failure_path = 'cleaned_file.csv' # Path to the heart_failure_dataset output file generated after isolation forest
    output_path = 'merged_dataset_regression.csv' # Path to the o/p file

    codebook, heart_failure = load_and_preprocess_data(codebook_path, heart_failure_path)
    merged_data = bin_based_concatentaion(codebook,heart_failure)
    
    merged_data.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
