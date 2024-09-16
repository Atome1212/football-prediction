from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils.scrap import download_csv, url_of_csv
from utils.update import update
from utils.modelling_soccer import get_df_from_db, just_train_model
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def scraper(**kwargs):
    """This function calls the url_of_csv and download_csv functions to scrape the csv files"""

    urls = url_of_csv()
    list_modified = {}

    if urls:
        print(f'{len(urls)} links found')
        for url in urls:
            returned_modified = download_csv(url)
            if returned_modified:
                if len(returned_modified[1]) > 0:
                    list_modified[returned_modified[0]] = returned_modified[1]
    else: 
        print("No link found")

    kwargs['ti'].xcom_push(key='csv_list_and_line', value=list_modified)

def update_db(**kwargs):
    """Update the database with the modified csv lines"""

    ti = kwargs['ti']
    csv_list_and_line = ti.xcom_pull(task_ids = 'scraper', key = 'csv_list_and_line')

    if csv_list_and_line:
        update(csv_list_and_line=csv_list_and_line)

def model_training():
    just_train_model(get_df_from_db())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes = 5),
}

with DAG(
    'football_scraper_dag',
    max_active_tasks = 1,
    default_args = default_args,
    description = 'DAG scraper football data',
    schedule_interval = timedelta(minutes = 1), # TODO : change to @daily
    start_date = datetime(2023, 1, 1),
    catchup = False,
) as dag:

    scraper_task = PythonOperator(
        task_id='scraper',
        python_callable=scraper
    )

    update_db_task = PythonOperator(
        task_id = 'update_db',
        python_callable=update_db
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=model_training
    )

    scraper_task >> update_db_task >> model_training_task
