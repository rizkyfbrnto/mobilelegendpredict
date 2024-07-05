import pandas as pd
import joblib

# Load datasets with corrected paths using raw string
cleaned_hero_data = pd.read_csv(r'data/cleaned_hero_data.csv')
corrected_heroes_cleaned = pd.read_csv(r'data/corrected_heroes_cleaned.csv')

# Convert specific columns to numeric values
def convert_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Convert columns to numeric
columns_to_convert = ['Winrate', 'Durability', 'Offense', 'Ability Effects', 'Difficulity']
corrected_heroes_cleaned = convert_to_numeric(corrected_heroes_cleaned, columns_to_convert)

# Pre-proses data dan simpan fitur
def preprocess_and_save_features():
    all_heroes_features = {}
    for hero in corrected_heroes_cleaned['Hero']:
        hero_data = corrected_heroes_cleaned[corrected_heroes_cleaned['Hero'] == hero].iloc[0]
        features = {feature: float(hero_data[feature]) if pd.notnull(hero_data[feature]) else 0.0 
                    for feature in ['Winrate', 'Durability', 'Offense', 'Ability Effects', 'Difficulity']}
        all_heroes_features[hero] = features

    joblib.dump(all_heroes_features, 'hero_features.pkl')

# Panggil pre-proses hanya sekali
preprocess_and_save_features()
