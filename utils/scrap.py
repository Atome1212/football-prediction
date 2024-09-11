import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from io import StringIO


def url_of_csv(url="https://www.football-data.co.uk/belgiumm.php", string_to_found="Jupiler League"):
    """That function return the url of the csv file of the league that we want to scrap"""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', string=string_to_found)
        return [f"https://{url.split('/')[2]}/{link.get('href')}" for link in links] if len(links) > 0 else False
    print(f"Erreur: {response.status_code}")
    return False

def download_csv(url):
    """That function downloads the csv file from the url and saves it in the csv folder with an index column"""

    response = requests.get(url)
    if response.status_code == 200:
        name = url.split("/")[-2]
        file_path = f'./csv/{name}.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            new_df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
            new_df.insert(0, 'index', range(1, len(new_df) + 1)) 
            
            if os.path.exists(file_path):
                current_df = pd.read_csv(file_path, on_bad_lines='skip')
                
                modified_indices = []
                for idx in range(min(len(new_df), len(current_df))):
                    if not new_df.iloc[idx].equals(current_df.iloc[idx]):
                        modified_indices.append(idx + 1)
                
                if len(new_df) > len(current_df):
                    new_rows_start = len(current_df)
                    new_rows = list(range(new_rows_start + 1, len(new_df) + 1)) 
                    modified_indices.extend(new_rows)

                if modified_indices:
                    os.remove(file_path)
                    new_df.to_csv(file_path, index=False)
                    print(f"{name} Updated. Modified rows: {modified_indices}")
                    return name, modified_indices
                else:
                    print(f"{name} already exists and is up-to-date")
                    return name, []

            else:
                new_df.to_csv(file_path, index=False)
                print(f"{name} Downloaded")
                return name, list(range(1, len(new_df) + 1))
                
        except pd.errors.EmptyDataError:
            print(f"{name} is empty or incorrectly formatted.")
        except pd.errors.ParserError as e:
            print(f"Error parsing {name}: {e}")
    else:
        print(f"Erreur: {response.status_code}")
    return False

def scraper():
    """This function calls the url_of_csv and download_csv functions to scrape the csv files"""
    
    urls = url_of_csv()
    if urls:
        print(f'{len(urls)} links found')
        list_modified = {}

        for url in urls:
            returned_modified = download_csv(url)
            if returned_modified:
                if len(returned_modified[1]) > 0:
                    list_modified[returned_modified[0]] = returned_modified[1]
                    
    else: 
        print("No link found")
        list_modified = {}


    print(list_modified)

    return list_modified 

def check_same_columns():
    """This function checks if the columns of the csv files are the same"""

    csv_files = os.listdir('./csv')
    columns = {}
    for file in csv_files:
        try:
            df = pd.read_fwf(f'./csv/{file}')
            for col in df.columns:
                if col not in columns:
                    columns[col] = 1
                else:
                    columns[col] += 1
            
        except pd.errors.ParserError as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")
    return columns

def main():
    scraper()
    print(len(check_same_columns()))    

if __name__ == '__main__':
    main()

