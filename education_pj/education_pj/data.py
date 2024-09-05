import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from bson import ObjectId

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
    # taker_irt_collection = db['3_candidateIRT']

    # 필요한 필드만 선택하여 데이터를 가져오는 함수
    def fetch_collection_as_dataframe(collection, fields):
        data = list(collection.find({}, fields))
        return pd.DataFrame(data)

    # 필요한 필드만 선택
    qa_fields = {'testID': 1, 'assessmentItemID': 1, 'learnerID': 1, 'answerCode': 1, '_id': 0}
    irt_fields = {'testID': 1, 'assessmentItemID': 1, 'knowledgeTag': 1, '_id': 0}

    # 데이터프레임으로 변환
    qa_df = fetch_collection_as_dataframe(qa_collection, qa_fields)
    irt_df = fetch_collection_as_dataframe(irt_collection, irt_fields)

    # 병합 전 필요한 컬럼만 유지 및 인덱스 설정
    qa_df = qa_df.set_index(['testID', 'assessmentItemID'])
    irt_df = irt_df.set_index(['testID', 'assessmentItemID'])

    # 최적화된 병합
    merged_df = pd.merge(qa_df, irt_df, left_index=True, right_index=True, how='left').reset_index()

    # 병합 후 필요한 컬럼만 선택
    merged_df = merged_df[['learnerID', 'answerCode', 'knowledgeTag']]
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')
    

    return merged_df  # 빈 데이터프레임 반환


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

    db_config = {
        'host': "10.41.2.78",
        'port': "5432",
        'user': "postgres",
        'pw': "qwe123",
        'db': "project",
        'table_name': "label_math_ele"
    }
    
    db_url = f"postgresql://{db_config['user']}:{db_config['pw']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    engine = create_engine(db_url)

    query = f"SELECT * FROM {db_config['table_name']}"
    label_math = pd.read_sql(query, engine)

    return label_math