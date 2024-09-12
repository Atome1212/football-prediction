import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_html_fixture():
    url = "https://www.walfoot.be/belgique/jupiler-pro-league/calendrier"
    response = requests.get(url)
    return response.text


def get_fixture():
    html = get_html_fixture()
    soup = BeautifulSoup(html, "html.parser")
    table2 = soup.find_all("tr", {"class": "table-active"})
    return table2

def get_fixture_data():
    table2 = get_fixture()
    results = []
    HomeTeam= []
    AwayTeam=[]
    dates = []

    for result in table2:
        results.append(result.find_all("td")[2].text)
    cleaned_results = [result.strip() for result in results]
    cleaned_results = cleaned_results[49:]

    for team in table2:
        HomeTeam.append(team.find_all("td")[1].text)
        AwayTeam.append(team.find_all("td")[3].text)  

    cleaned_HomeTeam = [team.strip() for team in HomeTeam]
    cleaned_AwayTeam = [team.strip() for team in AwayTeam]
    HomeTeam = cleaned_HomeTeam[49:]
    AwayTeam = cleaned_AwayTeam[49:]

    for team in table2:
        dates.append(team.find_all("td")[0].text)

    dates = dates[49:]
    dates = [date.split()[0] for date in dates]

    def add_year(date_str):
        day, month = date_str.split('/')
        #convert to integer
        month = int(month)
        year = '2024' if int(month) >= 7  else '2025'
        return f"{day}/{month}/{year}"

    dates = [add_year(x.split(" ")[0]) for x in dates]   

    return cleaned_results, HomeTeam, AwayTeam, dates

def get_fixture_df():

    results, HomeTeam, AwayTeam, dates = get_fixture_data()
    df = pd.DataFrame({"HomeTeam": HomeTeam, "Result": results, "AwayTeam": AwayTeam, "Date": dates})
    df.to_csv("future_matches.csv", index=False)
    return df

if __name__ == "__main__":
    print(get_fixture_df())

