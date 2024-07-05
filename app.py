from flask import Flask, render_template, request
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and preprocessed features
model = joblib.load('naive_bayes_model.pkl')
hero_features = joblib.load('hero_features.pkl')

# Load heroes data for interface
heroes_data = pd.read_csv(r'data/corrected_heroes_cleaned.csv')
heroes = heroes_data['Hero'].tolist()

@app.route('/')
def index():
    return render_template('index.html', heroes=heroes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        team_allies = [request.form['hero_allie_1'], request.form['hero_allie_2'], 
                       request.form['hero_allie_3'], request.form['hero_allie_4'], 
                       request.form['hero_allie_5']]
        team_enemy = [request.form['hero_enemy_1'], request.form['hero_enemy_2'], 
                      request.form['hero_enemy_3'], request.form['hero_enemy_4'], 
                      request.form['hero_enemy_5']]
        
        # Calculate features for both teams
        features_allies = calculate_team_features_from_input(team_allies)
        features_enemy = calculate_team_features_from_input(team_enemy)
        
        print(f"Features Allies: {features_allies}")  # Debug print to check input features
        print(f"Features Enemy: {features_enemy}")    # Debug print to check input features
        
        # Predict probability for team Allies
        prob_allies = model.predict_proba([features_allies])[0][1]
        # Calculate probability for team Enemy
        prob_enemy = 1 - prob_allies
        
        return render_template('index.html', heroes=heroes, 
                               prob_allies=f'{prob_allies * 100:.2f}%', 
                               prob_enemy=f'{prob_enemy * 100:.2f}%')
    except Exception as e:
        logging.exception("Error occurred during prediction")
        return render_template('index.html', heroes=heroes, error=str(e))

def calculate_team_features_from_input(team):
    features = {'Winrate': 0.0, 'Durability': 0.0, 'Offense': 0.0, 'Ability Effects': 0.0, 'Difficulity': 0.0}
    valid_heroes_count = 0

    for hero_name in team:
        if hero_name in hero_features:
            hero_data = hero_features[hero_name]
            valid_heroes_count += 1
            for feature in features:
                features[feature] += hero_data[feature]

    if valid_heroes_count > 0:
        for feature in features:
            features[feature] /= valid_heroes_count

    print(f"Team: {team} Features: {features}")  # Debug print to check the features

    return pd.Series(features)

if __name__ == '__main__':
    app.run(debug=True)
