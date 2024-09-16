import pandas as pd
from utils.db_connection import connect_to_db, close_connection

def fetch_match_id(cursor, csv_file, line):
    """Fetch the Match Id for the given CSV file and line"""
    sql = """
        SELECT `Match Id`
        FROM `csv_updates`
        WHERE `Csv File Name` = %s AND `Csv Line Number` = %s
    """
    cursor.execute(sql, (csv_file, line))
    return cursor.fetchone()

def find_or_create_team(cursor, connection, team_name):
    """Find a team by name or insert the team if not found"""
    find_teams = """SELECT `Team Id` FROM `teams` WHERE `Team Name` = %s;"""
    insert_team = """INSERT INTO `teams` (`Team Name`) VALUES (%s);"""
    
    cursor.execute(find_teams, (team_name,))
    team_id = cursor.fetchone()
    
    if team_id is None:
        cursor.execute(insert_team, (team_name,))
        connection.commit()
        cursor.execute(find_teams, (team_name,))
        team_id = cursor.fetchone()
    
    return team_id

def update_matches(cursor, home_team_id, away_team_id, date, match_id):
    """Update the matches table"""
    update_matches_sql = """
        UPDATE matches
        SET `Home Team Id` = %s,
            `Away Team Id` = %s,
            `Date` = %s
        WHERE `Match Id` = %s;
    """
    cursor.execute(update_matches_sql, (home_team_id, away_team_id, date, match_id))

def update_full_time_results(cursor, df, match_id):
    """Update the full_time_results table"""
    update_full_time_results_sql = """
        UPDATE full_time_results
            SET FTHG = %s,
                FTAG = %s,
                FTR = %s
            WHERE `Match Id` = %s;
    """
    cursor.execute(update_full_time_results_sql, (df['FTHG'], df['FTAG'], df['FTR'], match_id))

def update_half_time_results(cursor, df, match_id):
    """Update the half_time_results table"""
    update_half_time_results_sql = """
        UPDATE half_time_results
            SET HTHG = %s,
                HTAG = %s,
                HTR = %s
            WHERE `Match Id` = %s;
    """
    cursor.execute(update_half_time_results_sql, (df['HTHG'], df['HTAG'], df['HTR'], match_id))

def update_match_odds(cursor, df, match_id):
    """Update the match_odds table with new columns"""
    update_match_odds_sql = """
        UPDATE match_odds
            SET AvgH = %s,
                AvgA = %s,
                AHCh = %s,
                B365H = %s,
                B365D = %s,
                B365A = %s,
                BWH = %s,
                BWD = %s,
                BWA = %s,
                PSH = %s,
                PSD = %s,
                PSA = %s
            WHERE `Match Id` = %s;
    """
    cursor.execute(update_match_odds_sql, (df['AvgH'], df['AvgA'], df['AHCh'],df['B365H'], df['B365D'], df['B365A'],df['BWH'], df['BWD'], df['BWA'],df['PSH'], df['PSD'], df['PSA'], match_id))

def update_match_statistics(cursor, df, match_id):
    """Update the match_statistics table"""
    update_match_statistics_sql = """
        UPDATE match_statistics
        SET `HS` = %s,
            `AS` = %s,
            `HST` = %s,
            `AST` = %s,
            `HC` = %s,
            `AC` = %s,
            `HY` = %s,
            `AY` = %s,
            `HR` = %s,
            `AR` = %s
        WHERE `Match Id` = %s;
    """
    cursor.execute(update_match_statistics_sql, (df['HS'], df['AS'], df['HST'], df['AST'], df['HC'], df['AC'], df['HY'], df['AY'], df['HR'], df['AR'], match_id))

def process_csv_updates(connection, csv_list_and_line):
    """Process each CSV file and its modified lines"""
    with connection.cursor() as cursor:
        for csv_file, modified_lines in csv_list_and_line.items():
            for line in modified_lines:
                match_id = fetch_match_id(cursor, csv_file, line)
                
                if match_id:
                    df = pd.read_csv(f'./csv/{csv_file}.csv').loc[line-1]
                    
                    home_team_id = find_or_create_team(cursor, connection, df['HomeTeam'])
                    away_team_id = find_or_create_team(cursor, connection, df['AwayTeam'])
                    
                    update_matches(cursor, home_team_id, away_team_id, df['Date'], match_id[0])
                    update_full_time_results(cursor, df, match_id[0])
                    update_half_time_results(cursor, df, match_id[0])
                    update_match_odds(cursor, df, match_id[0])
                    update_match_statistics(cursor, df, match_id[0])
                    
                    print(f"Match Id {match_id[0]} updated")
                    connection.commit()

def update(**kwargs):
    """Main update function to fetch and update match statistics"""
    csv_list_and_line = kwargs.get('csv_list_and_line')

    if csv_list_and_line:
        connection = connect_to_db()
        try:
            process_csv_updates(connection, csv_list_and_line)
        except Exception as e:
            print(f"Error fetching Match Ids: {e}")
        finally:
            close_connection(connection)
    else:
        print("No csv file to fetch from the database")

def main():
    update()

if __name__ == '__main__':
    main()  
