import os
from datetime import datetime
import pandas as pd
from sqlalchemy import MetaData, Table, Column, Integer, String, Float
from db_connection import connect_to_db, close_connection
import pymysql

def create_table_dataset(cursor, table_name, list_of_columns, primary_keys=None):
    """Create a table in the database if it does not exist"""
    
    columns_definitions = []
    
    for element in list_of_columns:
        column_def = f"`{element[0]}` {element[1]}"
        if element[0] in primary_keys:
            if element[1] == 'INT': 
                column_def += " PRIMARY KEY AUTO_INCREMENT"
            else:
                column_def += " PRIMARY KEY"
        columns_definitions.append(column_def)
    
    columns = ", ".join(columns_definitions)
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
    
    print(query)
    cursor.execute(query)

def create_table_if_not_exists(connection, cursor, all_tables):
    """Create the tables if they do not exist"""
    for table_name, list_of_columns in all_tables.items():
        create_table_dataset(cursor, table_name, list_of_columns)

def initialization_table(connection, cursor, all_tables):
    """Create the tables if they do not exist"""
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
        create_table_dataset(cursor, table_name, list_of_columns, primary_keys.get(table_name, []))
    connection.commit()

def keep_only_last_6_csv_files(csv_directory):
    """Update the database with the last 6 csv files"""
    
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    csv_files.sort(reverse=True)
    
    print(f"CSV files found: {csv_files}")
    
    current_year = datetime.now().year % 100
    files_to_keep = []
    for year in range(current_year, current_year - 100, -1):
        year_str = f"{year:02d}"
        matching_files = [csv_file for csv_file in csv_files if csv_file.startswith(year_str)]
        files_to_keep.extend(matching_files)
        if len(files_to_keep) >= 6:
            return sorted(files_to_keep[:6])
    
    print(f"Files selected for processing: {files_to_keep}")
    return sorted(files_to_keep)


def get_existing_teams(cursor):
    query = "SELECT `Team Name` FROM teams"
    try:
        cursor.execute(query)
        existing_teams = cursor.fetchall()
        return [team['Team Name'] for team in existing_teams]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def insert_teams(connection, cursor, teams_df):
    """Insert new teams into the database"""
    
    query = "SELECT MAX(`Team Id`) as max_id FROM teams"
    
    cursor.execute(query)
    result = cursor.fetchone()
    current_max_id = result['max_id'] if result and result['max_id'] is not None else 0
    teams_df['Team Id'] = range(current_max_id + 1, current_max_id + 1 + len(teams_df))
    
    for _, row in teams_df.iterrows():
        team_id = row['Team Id'] if not pd.isna(row['Team Id']) else None
        team_name = row['Team Name'] if not pd.isna(row['Team Name']) else None

        cursor.execute(
            "INSERT INTO teams (`Team Id`, `Team Name`) VALUES (%s, %s)", 
            (team_id, team_name)
        )
    connection.commit()


def process_each_csv(connection, cursor, csv_directory, all_tables):
    """Process each csv file"""
    
    existing_teams = get_existing_teams(cursor)
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
                insert_teams(connection, cursor, temp_df)
                existing_teams.extend(new_teams)

            print("Match odds processed")
            process_csv_updates(connection, cursor, df, csv_file)
            print("Csv updates processed")
            process_matches(connection, cursor, df, all_tables['matches'])
            print("Matches processed")

            for i in ['half_time_results', 'full_time_results', 'match_statistics', 'match_odds']:
                process_other(connection, cursor, df, all_tables[i], i)
                print(f"{i} processed")
        else:
            print(f"File {full_path} does not exist.")


def process_matches(connection, cursor, df, tables):
    query = "SELECT MAX(`Match Id`) as max_id FROM matches"
    cursor.execute(query)
    result = cursor.fetchone()
    current_max_id = result['max_id'] if result and result['max_id'] is not None else 0

    for _, row in df.iterrows():
        home_team_id = get_team_id(cursor, row['HomeTeam'])
        away_team_id = get_team_id(cursor, row['AwayTeam'])
        match_id = current_max_id + 1
        current_max_id += 1

        home_team_id = home_team_id if not pd.isna(home_team_id) else None
        away_team_id = away_team_id if not pd.isna(away_team_id) else None
        match_date = row['Date'] if not pd.isna(row['Date']) else None

        cursor.execute(
            "INSERT INTO matches (`Match Id`, `Home Team Id`, `Away Team Id`, `Date`) VALUES (%s, %s, %s, %s)",
            (match_id, home_team_id, away_team_id, match_date)
        )

    connection.commit()

def process_other(connection, cursor, df, table, table_name):
    for _, row in df.iterrows():
        match_id = get_match_id(cursor, row['HomeTeam'], row['AwayTeam'], row['Date'])

        other = {'Match Id': match_id}
        for idx, i in enumerate(table):
            if idx == 0:
                continue
            other[i[0]] = row[i[0]] if not pd.isna(row[i[0]]) else None

        columns = ", ".join(f"`{col}`" for col in other.keys())
        values = ", ".join(["%s"] * len(other))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        
        cursor.execute(query, list(other.values()))
    connection.commit()

def get_team_id(cursor, team_name):
    if pd.isna(team_name):
        return None

    query = f"SELECT `Team Id` FROM teams WHERE `Team Name` = %s"
    cursor.execute(query, (team_name,))
    result = cursor.fetchone()
    return result['Team Id'] if result else None

def get_match_id(cursor, home_team, away_team, date):
    home_team = home_team if not pd.isna(home_team) else None
    away_team = away_team if not pd.isna(away_team) else None
    date = date if not pd.isna(date) else None

    query = """
    SELECT `Match Id` FROM matches
    WHERE `Home Team Id` = (SELECT `Team Id` FROM teams WHERE `Team Name` = %s)
    AND `Away Team Id` = (SELECT `Team Id` FROM teams WHERE `Team Name` = %s)
    AND `Date` = %s
    """
    
    cursor.execute(query, (home_team, away_team, date))
    result = cursor.fetchone()
    
    return result['Match Id'] if result else None

def process_csv_updates(connection, cursor, df, csv_file):
    for idx, row in df.iterrows():
        cursor.execute(
            "INSERT INTO csv_updates (`Match Id`, `Csv File Name`, `Csv Line Number`) VALUES (%s, %s, %s)",
            (0, csv_file.split('.')[0], idx)
        )
    connection.commit()

def db_creation():
    """Create the database"""
    connection = connect_to_db()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    all_tables = {
        'teams': [
            ('Team Id', 'INT'), 
            ('Team Name', 'VARCHAR(255)')
        ],
        'matches': [
            ('Match Id', 'INT'),
            ('Home Team Id', 'INT'), 
            ('Away Team Id', 'INT'), 
            ('Date', 'VARCHAR(255)')
        ],
        'full_time_results': [
            ('Match Id', 'INT'),
            ('FTHG', 'INT'),
            ('FTAG', 'INT'),
            ('FTR', 'VARCHAR(10)')
        ],
        'half_time_results': [
            ('Match Id', 'INT'),
            ('HTHG', 'INT'),
            ('HTAG', 'INT'),
            ('HTR', 'VARCHAR(10)') 
        ],
        'match_statistics': [
            ('Match Id', 'INT'),  
            ('HS', 'FLOAT'),  
            ('AS', 'FLOAT'), 
            ('HST', 'FLOAT'),  
            ('AST', 'FLOAT'),   
            ('HC', 'FLOAT'), 
            ('AC', 'FLOAT'), 
            ('HY', 'FLOAT'), 
            ('AY', 'FLOAT'),  
            ('HR', 'FLOAT'),  
            ('AR', 'FLOAT')   
        ],
        'match_odds': [
            ('Match Id', 'INT'), 
            ('AvgH', 'FLOAT'),
            ('AvgA', 'FLOAT'),
            ('AHCh', 'FLOAT'),
            ('B365H', 'FLOAT'),  
            ('B365D', 'FLOAT'), 
            ('B365A', 'FLOAT'),
            ('BWH', 'FLOAT'),
            ('BWD', 'FLOAT'),
            ('BWA', 'FLOAT'),
            ('PSH', 'FLOAT'),
            ('PSD', 'FLOAT'),
            ('PSA', 'FLOAT')
        ],
        'csv_updates': [
            ('Match Id', 'INT'), 
            ('Csv File Name', 'VARCHAR(255)'),
            ('Csv Line Number', 'INT')
        ]
    }

    initialization_table(connection, cursor, all_tables)
    process_each_csv(connection, cursor, 'csv', all_tables)

    close_connection(connection)

if __name__ == "__main__":
    db_creation()
