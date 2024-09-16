import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import streamlit as st
import base64

def fetch_match_data():
    url = 'https://www.walfoot.be/belgique/jupiler-pro-league/calendrier'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Failed to load page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.find_all('tr', class_='table-active')

def process_match_rows(match_rows):
    today = datetime.today()
    matches = []

    first_match = True
    for row in match_rows:
        if first_match:
            first_match = False
            continue

        date_time_elem = row.find('td', class_='text-center')
        if date_time_elem:
            date_time_text = date_time_elem.get_text(strip=True)

            try:
                date_str, time_str = date_time_text.split()
                day, month = map(int, date_str.split('/'))

                if 1 <= month <= 6:
                    year = 2025
                else:
                    year = 2024

                date_obj = datetime(year, month, day)

                team_1 = row.find_all('a')[0].get_text(strip=True)
                team_2 = row.find_all('a')[2].get_text(strip=True)

                matches.append({
                    'Date': date_obj.strftime("%Y-%m-%d"),
                    'Hometeam': team_1,
                    'Awayteam': team_2
                })

            except ValueError:
                print(f"Failed to process date: {date_time_text}")
    
    manual_matches = [
        {'Date': '2024-08-24', 'Hometeam': 'Cercle de Bruges', 'Awayteam': 'La Gantoise'},
        {'Date': '2024-08-24', 'Hometeam': 'Anderlecht', 'Awayteam': 'KRC Genk'}
    ]

    matches.extend(manual_matches)
    return matches

def remove_specific_matches(matches):
    matches_to_remove = [
        {'Hometeam': 'Anderlecht', 'Awayteam': 'KRC Genk', 'Date': '2024-09-17'},
        {'Hometeam': 'Cercle de Bruges', 'Awayteam': 'La Gantoise', 'Date': '2024-09-26'}
    ]

    matches = [match for match in matches if not any(
        match.get('Hometeam') == match_to_remove['Hometeam'] and
        match.get('Awayteam') == match_to_remove['Awayteam'] and
        match.get('Date') == match_to_remove['Date']
        for match_to_remove in matches_to_remove
    )]

    return matches

def assign_championship_day(matches):
    matches = remove_specific_matches(matches)
    matches.sort(key=lambda x: datetime.strptime(x['Date'], "%Y-%m-%d"))
    championship_day = 1
    match_counter = 0

    for match in matches:
        match['Championship Day'] = f"Championship Day {championship_day}"
        match_counter += 1

        if match_counter == 8:
            championship_day += 1
            match_counter = 0

    return matches

def filter_championship_days(matches):
    today = datetime.today()
    championship_days = {}

    for match in matches:
        day = match['Championship Day']
        match_date = datetime.strptime(match['Date'], "%Y-%m-%d")
        
        if day not in championship_days:
            championship_days[day] = {'matches': [], 'earliest_date': match_date}
        else:
            if match_date < championship_days[day]['earliest_date']:
                championship_days[day]['earliest_date'] = match_date

        championship_days[day]['matches'].append(match)

    filtered_days = {day: data for day, data in championship_days.items() if data['earliest_date'] >= today}

    filtered_matches = [data['matches'] for data in filtered_days.values()]
    
    return filtered_matches

team_conversion = {
    'KV Courtrai': 'Kortrijk',
    'Union SG': 'St. Gilloise',
    'La Gantoise': 'Gent',
    'FC Bruges': 'Club Brugge',
    'OH Louvain': 'Oud-Heverlee Leuven',
    'Beerschot': 'Beerschot VA',
    'KV Malines': 'Mechelen',
    'KRC Genk': 'Genk',
    'Charleroi': 'Charleroi',
    'Standard Liège': 'Standard',
    'STVV': 'St Truiden',
    'FCV Dender EH': 'Dender',
    'Antwerp': 'Antwerp',
    'Westerlo': 'Westerlo',
    'Cercle de Bruges': 'Cercle Brugge',
    'Anderlecht': 'Anderlecht'
}
def convert_team_name(team):
    return team_conversion.get(team, team)

def get_logo_base64(team):
    team = team_conversion.get(team, team)
    img_path = f"logos/{team}.png"
    try:
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return ""

def display_matches_app(filtered_matches):    
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        .match-info {
            text-align: center;
        }
        .button-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="centered">', unsafe_allow_html=True)
        
    st.title('Belgian Pro League Prediction App')

    st.image('logos/jupiler_pro_league.jpg', width=700)

    championship_days = sorted(
            list(set(
                match['Championship Day'] for match_group in filtered_matches for match in match_group
            )),
            key=lambda x: int(re.search(r'\d+', x).group())
        )

    selected_day = st.selectbox('Select a Championship Day:', championship_days)

    if selected_day:
        st.write(f"Matches for {selected_day}:")
            
        matches = [match for match_group in filtered_matches for match in match_group if match['Championship Day'] == selected_day]

        for match in matches:
            col1, col2, col3 = st.columns([5, 3, 2])
            with col1:
                st.write(f"{match['Hometeam']} vs {match['Awayteam']}")
            with col2:
                st.write(match['Date'])
            with col3:
                if st.button('Prediction', key=f"prediction_{match['Hometeam']}_{match['Awayteam']}"):
                    st.session_state.show_prediction = True
                    st.session_state.home_team = match['Hometeam']
                    st.session_state.away_team = match['Awayteam']
                    st.session_state.match_date = match['Date']
                    st.rerun()

def result(date, home_team, away_team):
    df = get_df_from_db()
    just_train_model(df)
    result_game = just_predict_match_result(df, date, home_team, away_team)
    return result_game

def show_prediction_page(home_team, away_team, date):
    home_team = convert_team_name(home_team)
    away_team = convert_team_name(away_team)

    prediction_result = result(date, home_team, away_team)

    st.markdown(
        """
        <style>
        .team-logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
        }
        .team-logo {
            max-height: 200px;
            max-width: 200px;
        }
        .vs-text {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            word-wrap: break-word;
            white-space: pre-line;
        }
        .return-button-container {
            display: flex;
            justify-content: center;
            margin-top: 50px;
        }
        .outcome-text-container {
            text-align: center;
            margin-top: 30px;
            font-size: 35px;
        }
        .championship-cup {
            font-size: 30px;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 15, 5])

    with col1:
        home_logo = get_logo_base64(home_team)
        if home_logo:
            st.markdown(f'<img class="team-logo" src="{home_logo}" alt="{home_team} logo">', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="vs-text">{home_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="vs-text">VS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="vs-text">{away_team}</div>', unsafe_allow_html=True)

    with col3:
        away_logo = get_logo_base64(away_team)
        if away_logo:
            st.markdown(f'<img class="team-logo" src="{away_logo}" alt="{away_team} logo">', unsafe_allow_html=True)

    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    st.markdown(f'''
        <div class="outcome-text-container">
            <span class="championship-cup">🏆</span>
            The predicted outcome of the game is: {prediction_result}
            <span class="championship-cup">🏆</span>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="return-button-container">', unsafe_allow_html=True)
    if st.button("Return"):
        st.session_state.show_prediction = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False

    if 'home_team' not in st.session_state:
        st.session_state.home_team = None

    if 'away_team' not in st.session_state:
        st.session_state.away_team = None

    if 'match_date' not in st.session_state:
        st.session_state.match_date = None

    match_rows = fetch_match_data()

    if match_rows:
        matches = process_match_rows(match_rows)
        matches = assign_championship_day(matches)
        matches = filter_championship_days(matches)

        if st.session_state.show_prediction:
            show_prediction_page(st.session_state.home_team, st.session_state.away_team, st.session_state.match_date)
        else:
            display_matches_app(matches)

if __name__ == "__main__":
    main()
