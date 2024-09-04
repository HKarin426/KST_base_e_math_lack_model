# extract.py

from pymongo import MongoClient
import pandas as pd

def get_mongo_connection(username, password, host, port, db_name):
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/')
    return client[db_name]

def fetch_collection_as_dataframe(collection):
    data = list(collection.find({}))
    return pd.DataFrame(data)

def extract_data():
    username = 'root'
    password = 'qwe123'
    host = '127.0.0.1'
    port = 27017
    db_name = 'project'

    db = get_mongo_connection(username, password, host, port, db_name)

    qa_collection = db['1_correct_answer']
    irt_collection = db['2_questionIRT']
    taker_irt_collection = db['3_candidateIRT']

    qa_df = fetch_collection_as_dataframe(qa_collection)
    irt_df = fetch_collection_as_dataframe(irt_collection)
    taker_irt_df = fetch_collection_as_dataframe(taker_irt_collection)

    return qa_df, irt_df, taker_irt_df
