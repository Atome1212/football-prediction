import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import SQLAlchemyError


def create_table_dataset(engine, table_name, metadata, list_of_columns, primary_keys=None):
    """Create a table in the database if it does not exist"""
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
    """Create the tables if they do not exist"""
    for table_name, list_of_columns in all_tables.items():
        create_table_dataset(engine, table_name, metadata, list_of_columns)              
        
def initialization_table(engine, all_tables):
    """Create the tables if they do not exist"""
    metadata = MetaData()
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
    """update the database with the last 6 csv files"""
    
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
    """Insert new teams into the database"""
    
    query = "SELECT MAX(`Team Id`) as max_id FROM teams"
    result = pd.read_sql(query, con=engine)
    current_max_id = result['max_id'].iloc[0] if not result.empty and result['max_id'].iloc[0] is not None else 0
    teams_df['Team Id'] = range(current_max_id + 1, current_max_id + 1 + len(teams_df))
    teams_df.to_sql('teams', con=engine, if_exists='append', index=False)

def process_each_csv(engine, csv_directory, all_tables):
    """Process each csv file"""
    
    existing_teams = get_existing_teams(engine)
    csv_files = keep_only_last_6_csv_files(csv_directory)
    for csv_file in csv_files:
        full_path = os.path.join(csv_directory, csv_file)
        # df1= pd.read_csv("future_matches.csv")
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

            print("Match odds processed")
            process_csv_updates(engine, df, csv_file)
            print("Csv updates processed")
            process_matches(engine, df, all_tables['matches'])
            print("Matches processed")

            for i in ['half_time_results', 'full_time_results', 'match_statistics', 'match_odds']:
                process_other(engine, df, all_tables[i], i)
                print(f"{i} processed")

        else:
            print(f"{full_path} does not exist.")

def process_matches(engine, df, tables):
    query = "SELECT MAX(`Match Id`) as max_id FROM matches"
    result = pd.read_sql(query, con=engine)
    current_max_id = result['max_id'].iloc[0] if not result.empty and result['max_id'].iloc[0] is not None else 0
    for index, row in df.iterrows():
        home_team_id = get_team_id(engine, row['HomeTeam'])
        away_team_id = get_team_id(engine, row['AwayTeam'])
        match_id = current_max_id + 1
        current_max_id += 1
        match_data = {
            'Match Id': match_id,
            'Home Team Id': home_team_id,
            'Away Team Id': away_team_id,
            'Date': row['Date']
        }
        match_df = pd.DataFrame([match_data])
        match_df.to_sql('matches', con=engine, if_exists='append', index=False)

def process_other(engine, df, table, table_name):
    for index, row in df.iterrows():
        match_id = get_match_id(engine, row['HomeTeam'], row['AwayTeam'], row['Date'])

        other = {'Match Id': match_id}

        for idx, i in enumerate(table):
            if idx == 0:
                continue
            other[i[0]] = row[i[0]]

        odds_df = pd.DataFrame([other])
        odds_df.to_sql(table_name, con=engine, if_exists='append', index=False)   

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

def process_csv_updates(engine, df, csv_file):
    csv_updates_data = {
        'Match Id': 0,
        'Csv File Name': csv_file.split('.')[0],
        'Csv Line Number': df["index"]
    }
    csv_updates_df = pd.DataFrame(csv_updates_data)
    csv_updates_df.to_sql('csv_updates', con=engine, if_exists='append', index=False)

# def future_matches(engine, df1):
#     for index, row in df1.iterrows():
#         home_team_id = get_team_id(engine, row['Home Team'])
#         away_team_id = get_team_id(engine, row['Away Team'])
        
        
        
#         future_matches_data = {
#             'Home Team Id': home_team_id,
#             'Result': row['Result'],
#             'Away Team Id': away_team_id,
#             'Date': row['Date']
#         }
        
#         future_matches_df = pd.DataFrame([future_matches_data])
#         future_matches_df.to_sql('future_matches', con=engine, if_exists='append', index=False)

def db_creation():
    """Create the database"""
    
    host = ''
    port = 3306
    user = ''
    password = ''
    database = ''
    csv_directory = 'csv'
    db_engine_str = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(db_engine_str)

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
            ('Match Id', Integer),
            ('FTHG', Integer),
            ('FTAG', Integer),
            ('FTR', String(10))
        ],
        'half_time_results': [
            ('Match Id', Integer),
            ('HTHG', Integer),
            ('HTAG', Integer),
            ('HTR', String(10)) 

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
            ('AvgH', Float),
            ('AvgA', Float),
            ('AHCh', Float),

            ('B365H', Float),  
            ('B365D', Float), 
            ('B365A', Float),    

            ('BWH', Float),
            ('BWD', Float),
            ('BWA', Float),

            ('PSH', Float),
            ('PSD', Float),
            ('PSA', Float)
        ],
        'csv_updates': [
            ('Match Id', Integer), 
            ('Csv File Name', String(255)),
            ('Csv Line Number', Integer)
        ],
        
        # 'future_matches': [
        #     ('Team Id', Integer), 
        #     ('Home Team Id', Integer), 
        #     ('Result', String(255)),
        #     ('Away Team Id', Integer),
        #     ('Date', String(255)),
            
        #     ]
    }

    initialization_table(engine, all_tables)
    process_each_csv(engine, csv_directory, all_tables)

if __name__ == "__main__":
    db_creation()
