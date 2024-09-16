
# ⚽ Football Prediction
<p align="center">
  <a href="https://www.football-data.co.uk">
      <img src="https://via.placeholder.com/500" alt="Football Prediction Logo" width="500" />
  </a>
</p>

## 🔣 Language & Tools
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org) [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![Airflow](https://img.shields.io/badge/airflow-%2300C7B7.svg?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org) [![MySQL](https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)

## 📝 Project Description

This project aims to predict the outcomes of football matches using a **Random Forest Classifier**. By scraping data from **football-data.co.uk**, we create datasets with relevant match statistics, train a model, and use it to predict upcoming football matches.

The project is structured into different modules for scraping data, processing and updating databases, training models, and making predictions. Data is organized and processed using **Airflow** DAGs to ensure timely updates, and predictions are made based on **team form**, **match statistics**, and **betting odds**.

## 📚 Table of Contents
- [📝 Project Description](#-project-description)
- [💻 Installation](#-installation)
- [🏃‍♂️ How to Run](#-how-to-run)
- [🗂️ Directory Structure](#-directory-structure)
- [🎓 Team Members](#-team-members)

## 💻 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Atome1212/football-prediction.git
    cd football-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Setup **MySQL** for the database:
   ```sql
    CREATE DATABASE football_prediction;
   ```
   Fill in the following fields in the db_creation.py script:
   ```bash
    host = 'localhost'   # MySQL server address (e.g., localhost)
    port = 3306          # MySQL port (usually 3306)
    user = 'root'        # MySQL username
    password = 'password'  # MySQL password
    database = 'football_prediction'  # Database name
   ```

   The necessary tables to store match data, statistics, and updates are defined in the script.
       The following tables will be created: <br>
            teams: Stores team information <br>
            matches: Stores match details <br>
            full_time_results: Stores full-time results of the match <br>
            half_time_results: Stores half-time results of the match <br>
            match_statistics: Stores match statistics (cards, corners, etc.) <br>
            match_odds: Stores betting odds for each match <br>
            csv_updates: Stores updated CSV lines <br>

  Run the script to create these tables:
  ```bash
    python db_creation.py
  ```
4. Install & Setup Airflow
   1. Install Airflow using pip:
      ```bash
        pip install apache-airflow=
      ```

   2. Initialize the Airflow database:
      ```bash
        airflow db init
      ```

   3. Create an Airflow user:
      ```bash
        airflow users create \
        --username admin \
        --firstname FIRST_NAME \
        --lastname LAST_NAME \
        --role Admin \
        --email admin@example.com
      ```
      
   4. Start the Airflow web server and scheduler:
      ```bash
        airflow webserver --port 8080
      ```
      and i an other terminal
      ```bash
        airflow scheduler
      ```




## 🏃‍♂️ How to Run

1. Run **Airflow** DAG to start data scraping and updating:
    ```bash
    airflow dags trigger football_scraper_dag
    ```

2. Use the **Random Forest Classifier** model for predictions:
    ```bash
    python modelling_soccer.py
    ```

3. For manual data scraping:
    ```bash
    python scrap.py
    ```

4. To update the database:
    ```bash
    python update.py
    ```

## 🗂️ Directory Structure

```bash
/football-prediction
├── dags
│   └── dag.py                         # Airflow DAG for scraping, updating DB, and training model
├── utils
│   ├── db_connection.py               # Database connection functions
│   ├── db_creation.py                 # Script for creating database tables
│   ├── scrap.py                       # Functions for scraping football data
│   ├── update.py                      # Functions for updating the database with new data
│   ├── modelling_soccer.py            # Football prediction model training and prediction
├── csv                                # Directory for storing scraped CSV files
├── models                             # Directory for storing trained models
└── requirements.txt                   # Python package dependencies
```

## 🎓 Team Members

- **👷‍♂️ [Atome1212](https://github.com/Atome1212)**: Data Engineer
- **👷‍♂️ [Siegfried2021](https://github.com/Siegfried2021)**: Data Engineer
- **👨‍💻 [ezgitandogan](https://github.com/ezgitandogan)**: Data Analyst
- **👩‍💻 [GeorginaAG](https://github.com/GeorginaAG)**: Data Analyst
