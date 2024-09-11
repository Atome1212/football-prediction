import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import SQLAlchemyError

def create_table_dataset(engine, table_name, metadata, list_of_columns, primary_keys=None):
    columns = [
        Column(element[0], element[1], primary_key=(element[0] in primary_keys)) 
        for element in list_of_columns
    ]
    
    dataset = Table(
        table_name, metadata,
        *columns,
        extend_existing=True
    )

    return dataset

def create_table_if_not_exists(engine, metadata, all_tables):
    for table_name, list_of_columns in all_tables.items():
        create_table_dataset(engine, table_name, metadata, list_of_columns)         
        
def initialization_table(engine):
    metadata = MetaData()
    all_tables = {
        'teams': [
            ('Team Id', Integer), 
            ('Team Name', String(255))
        ],
        'matches': [
            ('Match Id', Integer),
            ('Home Team Id', Integer), 
            ('Away Team Id', Integer), 
            ('Date', String(255))
        ],
        'full_time_results': [
            ('FTHG', Integer),
            ('FTAG', Integer),
            ('FTR', String(10)),
            ('Match Id', Integer)
        ],
        'half_time_results': [
            ('HTHG', Integer),
            ('HTAG', Integer),
            ('HTR', String(10)),
            ('Match Id', Integer)
        ],
        'match_statistics': [
            ('Match Id', Integer),  
            ('HS', Float),  
            ('AS', Float), 
            ('HST', Float),  
            ('AST', Float),   
            ('HC', Float), 
            ('AC', Float), 
            ('HY', Float), 
            ('AY', Float),  
            ('HR', Float),  
            ('AR', Float),   
        ],
        'match_odds': [
            ('Match Id', Integer), 
            ('B365H', Float),  
            ('B365D', Float), 
            ('B365A', Float),    

            ('WHH', Float),  
            ('WHD', Float),  
            ('WHA', Float)
        ],
        'csv_updates': [
            ('Match Id', Integer), 
            ('Csv File Name', String(255)),
            ('Csv Line Number', Integer)
        ]
    }
    
    primary_keys = {
        'teams': ['Team Id'],
        'matches': ['Match Id'],
        'full_time_results': ['Match Id'],
        'half_time_results': ['Match Id'],
        'match_statistics': ['Match Id'],
        'match_odds': ['Match Id'],
        'csv_updates': ['Match Id']
    }

    for table_name, list_of_columns in all_tables.items():
        create_table_dataset(engine, table_name, metadata, list_of_columns, primary_keys.get(table_name, []))
    metadata.create_all(engine)

def keep_only_last_6_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    csv_files.sort(reverse=True)  
    current_year = datetime.now().year % 100
    files_to_keep = []
    for year in range(current_year, current_year - 100, -1):
        year_str = f"{year:02d}"
        matching_files = [csv_file for csv_file in csv_files if csv_file.startswith(year_str)]
        files_to_keep.extend(matching_files)
        if len(files_to_keep) >= 6:
            return sorted(files_to_keep[:6])
    return sorted(files_to_keep)

def get_existing_teams(engine):
    query = "SELECT `Team Name` FROM teams"
    try:
        existing_teams = pd.read_sql(query, con=engine)
        return existing_teams['Team Name'].tolist()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def insert_teams(engine, teams_df):
    query = "SELECT MAX(`Team Id`) as max_id FROM teams"
    result = pd.read_sql(query, con=engine)
    current_max_id = result['max_id'].iloc[0] if not result.empty and result['max_id'].iloc[0] is not None else 0
    teams_df['Team Id'] = range(current_max_id + 1, current_max_id + 1 + len(teams_df))
    teams_df.to_sql('teams', con=engine, if_exists='append', index=False)

def process_each_csv(engine, csv_directory):
    existing_teams = get_existing_teams(engine)
    csv_files = keep_only_last_6_csv_files(csv_directory)
    for csv_file in csv_files:
        full_path = os.path.join(csv_directory, csv_file)
        if os.path.exists(full_path):
            print(f"Processing {csv_file}...")
            df = pd.read_csv(full_path)
            df.dropna(how='all', inplace=True)
            combined_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            new_teams = [team for team in combined_teams if team not in existing_teams]
            if new_teams:
                temp_df = pd.DataFrame({'Team Name': new_teams})
                insert_teams(engine, temp_df)
                existing_teams.extend(new_teams)

            process_csv_updates(engine, df, csv_file)
            print("Csv updates processed")
            process_matches(engine, df)
            print("Matches processed")
            process_full_time_results(engine, df)
            print("Full time results processed")
            process_half_time_results(engine, df)
            print("Half time results processed")
            process_match_statistics(engine, df)
            print("Match statistics processed")
            process_match_odds(engine, df)
            print("Match odds processed")

        else:
            print(f"{full_path} does not exist.")

def process_matches(engine, df):
    query = "SELECT MAX(`Match Id`) as max_id FROM matches"
    result = pd.read_sql(query, con=engine)
    current_max_id = result['max_id'].iloc[0] if not result.empty and result['max_id'].iloc[0] is not None else 0
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        home_team_id = get_team_id(engine, home_team)
        away_team_id = get_team_id(engine, away_team)
        match_id = current_max_id + 1
        current_max_id += 1
        match_data = {
            'Match Id': match_id,
            'Home Team Id': home_team_id,
            'Away Team Id': away_team_id,
            'Date': match_date
        }
        match_df = pd.DataFrame([match_data])
        match_df.to_sql('matches', con=engine, if_exists='append', index=False)

def get_team_id(engine, team_name):
    query = f"SELECT `Team Id` FROM teams WHERE `Team Name` = '{team_name}'"
    result = pd.read_sql(query, con=engine)
    return result['Team Id'].iloc[0] if not result.empty else None

def get_match_id(engine, home_team, away_team, date):
    query = f"""
    SELECT `Match Id` FROM matches
    WHERE `Home Team Id` = (SELECT `Team Id` FROM teams WHERE `Team Name` = '{home_team}')
    AND `Away Team Id` = (SELECT `Team Id` FROM teams WHERE `Team Name` = '{away_team}')
    AND `Date` = '{date}'
    """
    result = pd.read_sql(query, con=engine)
    return result['Match Id'].iloc[0] if not result.empty else None

def process_full_time_results(engine, df):
    for index, row in df.iterrows():
        match_id = get_match_id(engine, row['HomeTeam'], row['AwayTeam'], row['Date'])
        full_time_data = {
            'FTHG': row['FTHG'],
            'FTAG': row['FTAG'],
            'FTR': row['FTR'],
            'Match Id': match_id
        }
        full_time_df = pd.DataFrame([full_time_data])
        full_time_df.to_sql('full_time_results', con=engine, if_exists='append', index=False)

def process_half_time_results(engine, df):
    for index, row in df.iterrows():
        match_id = get_match_id(engine, row['HomeTeam'], row['AwayTeam'], row['Date'])

        half_time_data = {
            'HTHG': row['HTHG'],
            'HTAG': row['HTAG'],
            'HTR': row['HTR'],
            'Match Id': match_id
        }
        half_time_df = pd.DataFrame([half_time_data])
        half_time_df.to_sql('half_time_results', con=engine, if_exists='append', index=False)

def process_match_statistics(engine, df):
    for index, row in df.iterrows():
        match_id = get_match_id(engine, row['HomeTeam'], row['AwayTeam'], row['Date'])
        statistics_data = {
            'Match Id': match_id,
            'HS': row['HS'],
            'AS': row['AS'],
            'HST': row['HST'],
            'AST': row['AST'],
            'HC': row['HC'],
            'AC': row['AC'],
            'HY': row['HY'],
            'AY': row['AY'],
            'HR': row['HR'],
            'AR': row['AR'],
        }
        statistics_df = pd.DataFrame([statistics_data])
        statistics_df.to_sql('match_statistics', con=engine, if_exists='append', index=False)

def process_match_odds(engine, df):
    for index, row in df.iterrows():
        match_id = get_match_id(engine, row['HomeTeam'], row['AwayTeam'], row['Date'])

        odds_data = {
            'Match Id': match_id,
            'B365H': row['B365H'],
            'B365D': row['B365D'],
            'B365A': row['B365A'],
            'WHH': row['WHH'],
            'WHD': row['WHD'],
            'WHA': row['WHA']
        }
        odds_df = pd.DataFrame([odds_data])
        odds_df.to_sql('match_odds', con=engine, if_exists='append', index=False)   

def process_csv_updates(engine, df, csv_file):
    csv_updates_data = {
        'Match Id': 0,
        'Csv File Name': csv_file.split('.')[0],
        'Csv Line Number': df["index"]
    }
    csv_updates_df = pd.DataFrame(csv_updates_data)
    csv_updates_df.to_sql('csv_updates', con=engine, if_exists='append', index=False)

def db_creation():
    host = '/'
    port = 3306
    user = 'football-prediction'
    password = '1hjM!kuOX[rOmM_k'
    database = 'football-prediction'
    csv_directory = 'csv'
    db_engine_str = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(db_engine_str)
    initialization_table(engine)
    process_each_csv(engine, csv_directory)


if __name__ == "__main__":
    db_creation()
