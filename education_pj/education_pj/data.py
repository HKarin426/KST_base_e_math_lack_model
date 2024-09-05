import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine

# MongoDB 연결 설정
def fetch_mongo_data():
    username = 'root'
    password = 'qwe123'
    host = '10.41.2.78'
    port = 27017
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/')
    db = client['project']

    # 컬렉션 선택
    qa_collection = db['1_correct_answer']
    irt_collection = db['2_questionIRT']
    taker_irt_collection = db['3_candidateIRT']

    # 데이터를 데이터프레임으로 변환
    def fetch_collection_as_dataframe(collection):
        data = list(collection.find({}))
        return pd.DataFrame(data)

    # 데이터 병합
    qa_df = fetch_collection_as_dataframe(qa_collection)
    irt_df = fetch_collection_as_dataframe(irt_collection)
    taker_irt_df = fetch_collection_as_dataframe(taker_irt_collection)

    merged_df = pd.merge(qa_df, irt_df, on=['testID', 'assessmentItemID'], how='left')
    merged_df = pd.merge(merged_df, taker_irt_df, on=['learnerID', 'testID'], how='left')

    columns_to_drop = ['_id_x', 'Timestamp_x', '_id_y', 'Timestamp_y', '_id', 'Timestamp', 'learnerProfile_y']
    merged_df = merged_df.drop(columns=columns_to_drop)
    merged_df = merged_df.rename(columns={'learnerProfile_x': 'learnerProfile'})
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    return merged_df

# PostgreSQL 연결 설정
def fetch_postgres_data():
    db_config = {
        'host': "10.41.2.78",
        'port': "5432",
        'user': "postgres",
        'pw': "qwe123",
        'db': "project",
        'table_name': "education_2022"
    }

    db_url = f"postgresql://{db_config['user']}:{db_config['pw']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    engine = create_engine(db_url)

    query = f"SELECT * FROM {db_config['table_name']}"
    education_2022 = pd.read_sql(query, engine)

    education_2022 = education_2022.drop(columns=['Unnamed: 6'], errors='ignore')
    education_2022['계열화'] = education_2022['계열화'].fillna('정보 없음')

    return education_2022

# label_math 데이터 가져오기
def fetch_label_math():
    username = 'root'
    password = 'qwe123'
    host = '10.41.2.78'
    port = 27017
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/')
    db = client['project']
    collection = db['math_knowledge_data_set']

    data = list(collection.find({}))

    if not data:
        return pd.DataFrame()

    for entry in data:
        if '_id' in entry:
            entry['_id'] = str(entry['_id'])

    records = []
    for entry in data:
        for key, value in entry.items():
            if key.isdigit():
                from_concept = value.get('fromConcept', {})
                to_concept = value.get('toConcept', {})

                from_df = pd.json_normalize(from_concept, sep='_').add_prefix('from_')
                to_df = pd.json_normalize(to_concept, sep='_').add_prefix('to_')

                combined_df = pd.concat([from_df, to_df], axis=1)
                combined_df['id'] = key
                records.append(combined_df)

    final_df = pd.concat(records, ignore_index=True)
    return final_df
