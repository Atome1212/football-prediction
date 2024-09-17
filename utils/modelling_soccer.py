import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from datetime import datetime
from utils.db_connection import connect_to_db, close_connection

# Step1.1: Create a clean-and-preprocess object returning the uptaded df cleaned and including the engineered features

class FootballDataPreprocessor:
    """
    A class to preprocess football match data for analysis and feature engineering.
    
    Attributes:
        df (DataFrame): A copy of the original dataframe containing football match data.
    """
    def __init__(self, df):
        """
        Initializing the FootballDataPreprocessor class with a copy of the provided dataframe.
        
        Args:
            df (DataFrame): Original football match dataframe.
        """
        self.df = df.copy()

    def create_season(self):
        """
        Creates a 'season' column based on match dates.
        """
        if self.df["Date"].dtype != 'datetime64[ns]':
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

        bins = [
            pd.Timestamp('2019-07-01'),
            pd.Timestamp('2020-07-01'),
            pd.Timestamp('2021-07-01'),
            pd.Timestamp('2022-07-01'),
            pd.Timestamp('2023-07-01'),
            pd.Timestamp('2024-07-01'),
            pd.Timestamp('2025-07-01')
        ]
        season_labels = [
            '20192020',
            '20202021',
            '20212022',
            '20222023',
            '20232024',
            '20242025'
        ]
        
        self.df['season'] = pd.cut(self.df['Date'], bins=bins, labels=season_labels, right=False)

    def subset_df(self, cols_to_keep):
        """
        Subsets the dataframe to keep only the specified columns.
        
        Args:
            cols_to_keep (list): List of column names to retain in the dataframe.
        """
        self.df = self.df[cols_to_keep]

    def rewrite_date(self):
        """
        Rewrites the 'Date' column format from 'dd/mm/yyyy' or 'yyyy-mm-dd' to a standardized datetime format.
        """
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d/%m/%Y", errors='coerce')

        # Handle any remaining 'NaT' values by trying the second format
        self.df["Date"] = self.df["Date"].fillna(pd.to_datetime(self.df["Date"], format="%Y-%m-%d", errors='coerce'))

    def clean_data(self):
        """
        Removes rows with missing values (NaN) from the dataframe.
        """
        self.df = self.df.dropna()

    def label_column_result(self):
        """
        Adds a numeric result column ('FTR_num') based on the full-time result ('FTR').
        'H' (Home Win) is mapped to 1, 'D' (Draw) to 0, and 'A' (Away Win) to 2.
        """
        self.df['FTR_num'] = self.df['FTR'].map({'H':1, 'D':0, 'A':2})

    def window_rows_last_n_games(self, team, n_list):
        """
        Retrieves the last 'n' games played by a specified team, either home or away.
        
        Args:
            team (str): The team name.
            n_list (list): A list of integers specifying the number of games to retrieve.

        Returns:
            list: A list of dataframes containing the last 'n' games.
        """
        team_games = self.df[(self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)]
        team_games = team_games.sort_values(by="Date", ascending=False)

        return [team_games.head(n) for n in n_list]


    def window_rows_last_n_games_home(self, team, n_list):
        """
        Retrieves the last 'n' home games played by a specified team.
        
        Args:
            team (str): The team name.
            n_list (list): A list of integers specifying the number of games to retrieve.

        Returns:
            list: A list of dataframes containing the last 'n' home games.
        """
        team_games = self.df[self.df['HomeTeam'] == team]
        team_games = team_games.sort_values(by="Date", ascending=False)
        return [team_games.head(n) for n in n_list]

    def window_rows_last_n_games_away(self, team, n_list):
        """
        Retrieves the last 'n' away games played by a specified team.
        
        Args:
            team (str): The team name.
            n_list (list): A list of integers specifying the number of games to retrieve.

        Returns:
            list: A list of dataframes containing the last 'n' away games.
        """
        team_games = self.df[self.df['AwayTeam'] == team]
        team_games = team_games.sort_values(by="Date", ascending=False)
        return [team_games.head(n) for n in n_list]

    def window_rows_head_to_head(self, team1, team2):
        """
        Retrieves all head-to-head games between two specified teams.
        
        Args:
            team1 (str): Name of the first team.
            team2 (str): Name of the second team.

        Returns:
            DataFrame: A dataframe containing all head-to-head matches sorted by date.
        """
        teams_h2h = [team1, team2]
        h2h_df = self.df[(self.df['HomeTeam'].isin(teams_h2h)) & (self.df['AwayTeam'].isin(teams_h2h))].sort_values(by="Date", ascending=False)
        return h2h_df

    def compute_team_form(self, n_last_games, team):
        """
        Computes a form score for a team based on their last 'n' games.
        The score is based on results from the current and previous seasons, with different weights.

        Args:
            n_last_games (DataFrame): Dataframe containing the team's last 'n' games.
            team (str): The team name.

        Returns:
            float: The computed form score.
        """
        score = 0
        current_season = n_last_games.iloc[0]["season"]
        for idx, row in n_last_games.iloc[1:].iterrows():
            if row["season"] == current_season:
                score += 1 if (team == row['HomeTeam'] and row['FTR'] == 'H') or (team == row['AwayTeam'] and row['FTR'] == 'A') else 0.5 if row['FTR'] == 'D' else 0
            else:
                score += 0.5 if (team == row['HomeTeam'] and row['FTR'] == 'H') or (team == row['AwayTeam'] and row['FTR'] == 'A') else 0.25 if row['FTR'] == 'D' else 0
        return score
    
    def compute_team_form_home(self, n_last_games_home):
        """
        Computes a form score for a team based on their last 'n' home games.
        
        Args:
            n_last_games_home (DataFrame): Dataframe containing the team's last 'n' home games.

        Returns:
            float: The computed home form score.
        """
        home_score = 0
        current_season = n_last_games_home.iloc[0]["season"]
        for idx, row in n_last_games_home.iloc[1:].iterrows():
            if row["season"] == current_season:
                home_score += 1 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0
            else:
                home_score += 0.5 if row['FTR'] == 'H' else 0.25 if row['FTR'] == 'D' else 0
        return home_score
    
    def compute_team_form_away(self, n_last_games_away):
        """
        Computes a form score for a team based on their last 'n' away games.
        
        Args:
            n_last_games_away (DataFrame): Dataframe containing the team's last 'n' away games.

        Returns:
            float: The computed away form score.
        """
        away_score = 0
        current_season = n_last_games_away.iloc[0]["season"]
        for idx, row in n_last_games_away.iloc[1:].iterrows():
            if row["season"] == current_season:
                away_score += 1 if row['FTR'] == 'A' else 0.5 if row['FTR'] == 'D' else 0
            else:
                away_score += 0.5 if row['FTR'] == 'A' else 0.25 if row['FTR'] == 'D' else 0
        return away_score

    def compute_team_stats(self, n_last_games, team):
        """
        Computes aggregate statistics (yellow cards, red cards, shots, goals, etc.) for a team's last 'n' games.
        The stats are averaged, and games from past seasons are weighted less.

        Args:
            n_last_games (DataFrame): Dataframe containing the team's last 'n' games.
            team (str): The team name.

        Returns:
            list: A list of averaged team statistics over the last 'n' games.
        """
        if len(n_last_games) <= 1:
            return [0] * 6
        
        def sum_stats(row, is_home, factor=1):
            return [row[col] / factor for col in (['HY', 'HR', 'HS', 'HST', 'HC', 'FTHG'] if is_home else ['AY', 'AR', 'AS', 'AST', 'AC', 'FTAG'])]
        
        stats = [0] * 6
        current_season = n_last_games.iloc[0]['season']
        for idx, row in n_last_games.iloc[1:].iterrows():
            factor = 2 if row['season'] != current_season else 1
            stats = [a + b for a, b in zip(stats, sum_stats(row, row['HomeTeam'] == team, factor))]
        return [stat / (len(n_last_games) - 1) for stat in stats]
        
    def compute_h2h_score(self, h2h_df, hometeam):
        """
        Computes a score for a team's head-to-head performance against another team.
        The score is based on wins, losses, and draws between the two teams.
        
        Args:
            h2h_df (DataFrame): Dataframe containing the head-to-head match data.
            hometeam (str): Name of the home team for which to compute the score.

        Returns:
            float: The computed head-to-head score.
        """
        score = sum(1 if row['HomeTeam'] == hometeam and row['FTR'] == 'H' else -1.5 if row['HomeTeam'] == hometeam and row['FTR'] == 'A' else -1 if row['FTR'] == 'H' else 1.5 if row["FTR"] == "A" else 0 for idx, row in h2h_df.iloc[1:].iterrows())
        return score / len(h2h_df) if len(h2h_df) else 0
    
    def add_features(self, n_list):
        """
        Adds various form and statistical features for each team in the dataset, such as form, averages for yellow cards, red cards, goals, etc.
        
        Args:
            n_list (list): A list of integers specifying the number of last games to consider for each feature.
            
        Returns:
            DataFrame: The updated dataframe with new features added.
        """
        for index, row in self.df.iterrows():
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            n_last_games_hometeam = self.window_rows_last_n_games(home_team, n_list)
            n_last_games_awayteam = self.window_rows_last_n_games(away_team, n_list)
            n_last_games_home = self.window_rows_last_n_games_home(home_team, n_list)
            n_last_games_away = self.window_rows_last_n_games_away(away_team, n_list)
            h2h_df = self.window_rows_head_to_head(home_team, away_team)

            for i, n in enumerate(n_list):
                self.df.loc[index, f'home_team_form{n}'] = self.compute_team_form(n_last_games_hometeam[i], home_team)
                self.df.loc[index, f'away_team_form{n}'] = self.compute_team_form(n_last_games_awayteam[i], away_team)
                self.df.loc[index, f'home_team_form_home{n}'] = self.compute_team_form_home(n_last_games_home[i])
                self.df.loc[index, f'away_team_form_away{n}'] = self.compute_team_form_away(n_last_games_away[i])
                self.df.loc[index, [f'home_avg_yellow{n}', f'home_avg_red{n}', f'home_avg_shots{n}', f'home_avg_target{n}', f'home_avg_corners{n}', f'home_avg_goals{n}']] = self.compute_team_stats(n_last_games_hometeam[i], home_team)
                self.df.loc[index, [f'away_avg_yellow{n}', f'away_avg_red{n}', f'away_avg_shots{n}', f'away_avg_target{n}', f'away_avg_corners{n}', f'away_avg_goals{n}']] = self.compute_team_stats(n_last_games_awayteam[i], away_team)

            
            self.df.loc[index, 'h2h_record'] = self.compute_h2h_score(h2h_df, home_team)
        self.df['home_away_ratio'] = self.df['AvgH'] / self.df['AvgA']
        return self.df

    def subset_clean_rewrite(self):
        """
        A helper function that subsets the dataframe, adds a 'season' column, cleans the data, and rewrites the date format.
        
        Returns:
            DataFrame: The cleaned and subset dataframe.
        """
        self.create_season()
        cols_to_keep = ['Date', 'season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                        'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgA', 
                        'AHCh', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']
        self.subset_df(cols_to_keep)
        self.clean_data()
        return self.df

    def full_process(self):
        self.subset_clean_rewrite()  
        print(self.df.head()) 
        self.add_features([6, 8, 10, 12, 14])
        self.label_column_result()
        return self.df
        
    def add_new_row(self, date, home_team, away_team, n_list):
        """
        Adds a new row to the DataFrame with the provided match information (date, home team, away team),
        computes additional features based on the last n games for each team, and imputes missing values
        for specific columns using KNN. 
        Ensures that the new row is not added if it already exists in the DataFrame.
    
        Args:
            date (str or datetime): The date of the match to be added.
            home_team (str): The name of the home team for the match.
            away_team (str): The name of the away team for the match.
            n_list (list of int): The list specifying how many last games to consider when calculating aggregate features 
        
        Returns:
            DataFrame: The updated DataFrame with the new row added, computed features, and missing values imputed.
    
        """
        # Attempt to parse the date with multiple formats
        try:
            parsed_date = pd.to_datetime(date, format="%d/%m/%Y", errors='raise')
        except ValueError:
            parsed_date = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce')
        
        if not self.df[(self.df['Date'] == parsed_date) & (self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)].empty:
            return self.df
        
        # Proceed to add new row
        new_row = pd.DataFrame([{'Date': parsed_date, 'HomeTeam': home_team, 'AwayTeam': away_team}])
        self.df = pd.concat([new_row, self.df], ignore_index=True)        
        self.rewrite_date()
        self.df = self.df.sort_values(by="Date", ascending=False).reset_index(drop=True)
        self.create_season()
        new_row_index = self.df[(self.df['Date'] == parsed_date) & (self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)].index
    
        # Compute features for the new row (no changes to this part)
        n_last_games_hometeam = self.window_rows_last_n_games(home_team, n_list)
        n_last_games_awayteam = self.window_rows_last_n_games(away_team, n_list)
        n_last_games_home = self.window_rows_last_n_games_home(home_team, n_list)
        n_last_games_away = self.window_rows_last_n_games_away(away_team, n_list)
        h2h_df = self.window_rows_head_to_head(home_team, away_team)
    
        for i, n in enumerate(n_list):
            self.df.loc[new_row_index, f'home_team_form{n}'] = self.compute_team_form(n_last_games_hometeam[i], home_team)
            self.df.loc[new_row_index, f'away_team_form{n}'] = self.compute_team_form(n_last_games_awayteam[i], away_team)
            self.df.loc[new_row_index, f'home_team_form_home{n}'] = self.compute_team_form_home(n_last_games_home[i])
            self.df.loc[new_row_index, f'away_team_form_away{n}'] = self.compute_team_form_away(n_last_games_away[i])
    
            home_stats = self.compute_team_stats(n_last_games_hometeam[i], home_team)
            self.df.loc[new_row_index, f'home_avg_yellow{n}'] = home_stats[0]
            self.df.loc[new_row_index, f'home_avg_red{n}'] = home_stats[1]
            self.df.loc[new_row_index, f'home_avg_shots{n}'] = home_stats[2]
            self.df.loc[new_row_index, f'home_avg_target{n}'] = home_stats[3]
            self.df.loc[new_row_index, f'home_avg_corners{n}'] = home_stats[4]
            self.df.loc[new_row_index, f'home_avg_goals{n}'] = home_stats[5]
    
            away_stats = self.compute_team_stats(n_last_games_awayteam[i], away_team)
            self.df.loc[new_row_index, f'away_avg_yellow{n}'] = away_stats[0]
            self.df.loc[new_row_index, f'away_avg_red{n}'] = away_stats[1]
            self.df.loc[new_row_index, f'away_avg_shots{n}'] = away_stats[2]
            self.df.loc[new_row_index, f'away_avg_target{n}'] = away_stats[3]
            self.df.loc[new_row_index, f'away_avg_corners{n}'] = away_stats[4]
            self.df.loc[new_row_index, f'away_avg_goals{n}'] = away_stats[5]
    
        self.df.loc[new_row_index, 'h2h_record'] = self.compute_h2h_score(h2h_df, home_team)
    
        imputer = KNNImputer(n_neighbors=5)
        betting_cols_to_impute = ['AvgH', 'AvgA', 'AHCh', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']
        self.df[betting_cols_to_impute] = imputer.fit_transform(self.df[betting_cols_to_impute])
        self.df.loc[new_row_index, 'home_away_ratio'] = self.df.loc[new_row_index, 'AvgH'] / self.df.loc[new_row_index, 'AvgA']
    
        return self.df
   
    
# Step1.2: Find the find combination of n games sequences to optimize model performance (working on the cleaned and preprocessed df)

def run_model_experiments(updated_df, n_combinations=35000, target_col='FTR_num'):
    # Constant columns to include in every feature set
    constant_cols = ['season', 'h2h_record', 'home_away_ratio', 'AHCh', 'B365H', 'B365D', 'B365A',
                     'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']

    # Possible values for n (related to the number of rows included in the previous game sequences used for feature engineering)
    n_values = [6, 8, 10, 12, 14]

    # Aggregate features with variable number of game sequences
    features_aggregated = [
        'home_team_form', 'away_team_form', 'home_team_form_home', 'away_team_form_away',
        'home_avg_yellow', 'home_avg_red', 'home_avg_shots', 'home_avg_target', 'home_avg_corners', 'home_avg_goals',
        'away_avg_yellow', 'away_avg_red', 'away_avg_shots', 'away_avg_target', 'away_avg_corners', 'away_avg_goals'
    ]

    # Function to randomly select one n for each aggregate feature
    def get_random_feature_set(features_aggregated, n_values):
        return [f'{feature}{random.choice(n_values)}' for feature in features_aggregated]

    # Evaluation of the random forest classifier model
    def evaluation_model(df, features, target_col):
        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return accuracy_score(y_test, predictions)

    # Function to return and train a model on each combination of features with variable games sequences n
    def process_single_combination(i, updated_df, features_aggregated, n_values):
        # Generating a random feature set
        random_feature_set = get_random_feature_set(features_aggregated, n_values)

        # Combine constant columns and the randomly generated set of aggregate features
        total_features = constant_cols + random_feature_set

        # Check if all selected features exist in the DataFrame
        available_features = [f for f in total_features if f in updated_df.columns]
        
        # Skip this combination if not all features exist
        if len(available_features) < len(total_features):
            return None

        # Evaluate the model for the available features
        accuracy = evaluation_model(updated_df, available_features, target_col)
        
        return (random_feature_set, accuracy)

    # Parallel processing to evaluate multiple combinations of features
    results = Parallel(n_jobs=-1)(delayed(process_single_combination)(i, updated_df, features_aggregated, n_values) for i in range(n_combinations))

    # Filter out None values in the results (in case some combinations had missing features)
    results = [result for result in results if result is not None]

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results, columns=['final_feature_set', 'accuracy'])

    # Sort results by accuracy to find the best combinations
    sorted_results = results_df.sort_values(by='accuracy', ascending=False)

    # Get the best feature set and its accuracy score
    best_feature_set = sorted_results.iloc[0]['final_feature_set']

    return best_feature_set

# Step1.3: Train the model on the best feature set

def train_model_with_best_features(df, best_feature_set, constant_cols, target_col='FTR_num'):
    total_features = constant_cols + best_feature_set
    
    X = df[total_features]
    y = df[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Initialize the RandomForestClassifier model
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    best_model.fit(X_train, y_train)

    # Save the model with the best feature set
    date_time_now = datetime.now().strftime('%Y%m%d%H%M%S')


    os.makedirs(f'./models/{date_time_now}', exist_ok=True)

    joblib.dump(best_model, f"./models/{date_time_now}/best_random_forest_model.joblib")
    
    # Save the total features (constant + best features)
    joblib.dump(total_features, f"./models/{date_time_now}/best_feature_set.joblib")

    # Return accuracy on the test set for validation
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

# Step1.4: Predicting the result of the next match
def predict_match_result(new_df, model_path, constant_cols):
    # Load the pre-trained model
    loaded_model = joblib.load(model_path)
    
    # Load the total feature set used during training
    best_feature_set = joblib.load(model_path.replace('best_random_forest_model.joblib', 'best_feature_set.joblib'))

    # Selecting the first row of the appended df for prediction
    match_for_prediction = new_df.head(1)
    
    # Use the same features as during training
    match_for_prediction_features = match_for_prediction[best_feature_set]

    # Making the prediction and mapping to a readable outcome
    predicted_result = loaded_model.predict(match_for_prediction_features)
    result_map = {1: 'Home Win', 0: 'Draw', 2: 'Away Win'}
    
    return result_map[predicted_result[0]]

def just_train_model(df):
    # Step1: Preprocess the football data
    preprocessor_df = FootballDataPreprocessor(df)
    updated_df = preprocessor_df.full_process()

    # Step : Find the best feature set for the model
    best_feature_set = run_model_experiments(updated_df, n_combinations=35000, target_col='FTR_num')
    
    # List of constant features used in the model
    constant_cols = ['season', 'h2h_record', 'home_away_ratio', 'AHCh', 'B365H', 'B365D', 'B365A',
                     'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']
    
    # Step3: Train the RandomForest model with the best feature set
    train_model_with_best_features(updated_df, best_feature_set, constant_cols, target_col='FTR_num')

def just_predict_match_result(df, date, home_team, away_team):
    # Step1: Preprocess the football data
    preprocessor_df = FootballDataPreprocessor(df)
    
    # List of constant features used in the model
    constant_cols = ['season', 'h2h_record', 'home_away_ratio', 'AHCh', 'B365H', 'B365D', 'B365A',
                     'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']

    # Step2: Add the new match row and compute the relevant features for both teams (home and away)
    updated_df_2 = preprocessor_df.add_new_row(date, home_team, away_team, [6, 8, 10, 12, 14])

    # Step3: Find the trained model file

    folders = [f for f in os.listdir('./models') if os.path.isfile(os.path.join('./models', f, "best_random_forest_model.joblib"))]

    if folders:
        oldest_folder = "./models/" + max(folders, key=lambda f: extract_date_from_filename(f)) + "/best_random_forest_model.joblib"
        print(f"Le fichier le plus ancien est : {oldest_folder}")
    else:
        print("Aucun fichier .joblib trouvé dans le dossier.")
        return None  # If no model is found, exit the function
    
    # Step4: Predict the result of the new match
    result_game = predict_match_result(updated_df_2, oldest_folder, constant_cols)

    return result_game


# Step_all: Combined preprocessor and model selection
def preprocessing_and_model_selection(df, date, home_team, away_team):
    # Step1: Preprocess the football data
    preprocessor_df = FootballDataPreprocessor(df)
    updated_df = preprocessor_df.full_process()

    # Step : Find the best feature set for the model
    best_feature_set = run_model_experiments(updated_df, n_combinations=35000, target_col='FTR_num')
    
    # List of constant features used in the model
    constant_cols = ['season', 'h2h_record', 'home_away_ratio', 'AHCh', 'B365H', 'B365D', 'B365A',
                     'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA']
    
    # Step3: Train the RandomForest model with the best feature set
    train_model_with_best_features(updated_df, best_feature_set, constant_cols, target_col='FTR_num')

    # Step4: Add the new match row and compute the relevant features
    updated_df_2 = preprocessor_df.add_new_row(date, home_team, away_team, [6, 8, 10, 12, 14])

    # Step5: Predict the result of the new match

    folders = [f for f in os.listdir('./models') if os.path.isfile(os.path.join('./models', f, "best_random_forest_model.joblib"))]
    print(folders)

    if folders:
        oldest_folder = "./models/" + max(folders, key=lambda f: extract_date_from_filename(f) + "/best_random_forest_model.joblib")
        print(f"Le fichier le plus ancien est : {oldest_folder}")
    else:
        print("Aucun fichier .joblib trouvé dans le dossier.")
        return None  # If no model is found, exit the function
    
    # Step4: Predict the result of the new match
    result_game = predict_match_result(updated_df_2, oldest_folder, constant_cols)

    return result_game

def extract_date_from_filename(folder_name):
    return datetime.strptime(folder_name, '%Y%m%d%H%M%S')


def get_df_from_db():
    connection = connect_to_db()
    cursor = connection.cursor()    
    cursor.execute("""
        SELECT 
            matches.`Match Id`,
            homeTeam.`Team Name` AS HomeTeam,
            awayTeam.`Team Name` AS AwayTeam,
            DATE_FORMAT(STR_TO_DATE(matches.`Date`, '%d/%m/%Y'), '%Y-%m-%d') AS Date,
            full_time_results.`FTHG`,
            full_time_results.`FTAG`,
            full_time_results.`FTR`,
            half_time_results.`HTHG`,
            half_time_results.`HTAG`,
            half_time_results.`HTR`,
            match_statistics.`HS`,
            match_statistics.`AS`,
            match_statistics.`HST`,
            match_statistics.`AST`,
            match_statistics.`HC`,
            match_statistics.`AC`,
            match_statistics.`HY`,
            match_statistics.`AY`,
            match_statistics.`HR`,
            match_statistics.`AR`,
            match_odds.`AvgH`,
            match_odds.`AvgA`,
            match_odds.`AHCh`,
            match_odds.`B365H`,
            match_odds.`B365D`,
            match_odds.`B365A`,
            match_odds.`BWH`,
            match_odds.`BWD`,
            match_odds.`BWA`,
            match_odds.`PSH`,
            match_odds.`PSD`,
            match_odds.`PSA`
        FROM matches
        JOIN teams AS homeTeam ON homeTeam.`Team Id` = matches.`Home Team Id`
        JOIN teams AS awayTeam ON awayTeam.`Team Id` = matches.`Away Team Id`
        JOIN full_time_results ON matches.`Match Id` = full_time_results.`Match Id`
        JOIN half_time_results ON matches.`Match Id` = half_time_results.`Match Id`
        JOIN match_statistics ON matches.`Match Id` = match_statistics.`Match Id`
        JOIN match_odds ON matches.`Match Id` = match_odds.`Match Id`
        JOIN csv_updates ON matches.`Match Id` = csv_updates.`Match Id`
        ORDER BY matches.`Date` DESC        
    """)

    rows = cursor.fetchall()
    close_connection(connection)
    df = pd.DataFrame.from_records(rows, columns=[x[0] for x in cursor.description])
    return df

# Main function
def main():
    df = get_df_from_db() 
    date = '23/09/2024'
    home_team = 'St. Gilloise'
    away_team = 'Anderlecht'

    just_train_model(df)
    result_game = just_predict_match_result(df, date, home_team, away_team)

    print(result_game)

if __name__ == "__main__":
    main()
