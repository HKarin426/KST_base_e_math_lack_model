import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, text

# DB Configuration
def load_qa_irt_theta_data():
    # MongoDB 연결
    username = 'root'
    password = 'qwe123'
    host = '10.41.2.78'
    port = 27017
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/')
    db = client['project']

    # Collections
    qa_collection = db['1_correct_answer']
    irt_collection = db['2_questionIRT']
    theta_collection = db['3_candidateIRT']

    # 필드 설정
    qa_fields = {'testID': 1, 'assessmentItemID': 1, 'learnerID': 1, 'answerCode': 1, '_id': 0}
    irt_fields = {
        'testID': 1,
        'assessmentItemID': 1,
        'knowledgeTag': 1,
        'difficultyLevel': 1,
        'discriminationLevel': 1,
        'guessLevel': 1,
        '_id': 0
    }
    theta_fields = {
        'testID': 1,
        'learnerID': 1,
        'theta': 1,
        '_id': 0
    }

    # Convert to DataFrame
    qa_df = pd.DataFrame(list(qa_collection.find({}, qa_fields)))
    irt_df = pd.DataFrame(list(irt_collection.find({}, irt_fields)))
    theta_df = pd.DataFrame(list(theta_collection.find({}, theta_fields)))

    # Merge DataFrames
    qa_df = qa_df.set_index(['testID', 'assessmentItemID'])
    irt_df = irt_df.set_index(['testID', 'assessmentItemID'])
    theta_df = theta_df.set_index(['testID', 'learnerID'])

    merged_df = pd.merge(qa_df, irt_df, left_index=True, right_index=True, how='left')
    merged_df = pd.merge(merged_df, theta_df, left_on=['testID', 'learnerID'], right_on=['testID', 'learnerID'], how='left').reset_index()

    # Select relevant columns
    merged_df = merged_df[['learnerID', 'answerCode', 'knowledgeTag', 'difficultyLevel', 'discriminationLevel', 'guessLevel', 'theta']]
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    return merged_df

def load_label_math_ele():
    db_config = {
        'host': "10.41.2.78",
        'port': "5432",
        'user': "postgres",
        'pw': "qwe123",
        'db': "project",
        'table_name': "label_math_ele"
    }

    # PostgreSQL connection
    db_url = f"postgresql://{db_config['user']}:{db_config['pw']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    engine = create_engine(db_url)

    # Fetch the table
    query = f"SELECT * FROM {db_config['table_name']}"
    label_math_ele = pd.read_sql(query, engine)

    return label_math_ele

def load_education_data():
    db_config = {
        'host': "10.41.2.78",
        'port': "5432",
        'user': "postgres",
        'pw': "qwe123",
        'db': "project",
        'table_name': "education_2022"
    }

    # PostgreSQL connection
    db_url = f"postgresql://{db_config['user']}:{db_config['pw']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    engine = create_engine(db_url)

    # Fetch the table
    query = f"SELECT * FROM {db_config['table_name']}"
    education_2022 = pd.read_sql(query, engine)

    # Clean the data
    education_2022 = education_2022.drop(columns=['Unnamed: 6'], errors='ignore')
    education_2022['계열화'] = education_2022['계열화'].fillna('정보 없음')

    return education_2022

def process_merged_data():
    # Load data
    merged_df = load_qa_irt_theta_data()  # MongoDB 데이터 (qa, irt, theta)
    education_2022 = load_education_data()  # PostgreSQL 데이터 (education_2022)

    # Ensure ID column in education_2022 is numeric to match with merged_df's knowledgeTag
    education_2022['ID'] = pd.to_numeric(education_2022['ID'], errors='coerce', downcast='integer')

    # Convert knowledgeTag in merged_df to numeric, ensuring no leading/trailing whitespaces
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    # Check for common values between merged_df['knowledgeTag'] and education_2022['ID']
    common_ids = merged_df['knowledgeTag'].isin(education_2022['ID'])

    # If there are no common values, exit the function
    if common_ids.sum() == 0:
        print("No common values between knowledgeTag and ID. Check the source data.")
        return merged_df

    # Merge education_2022 with merged_df on knowledgeTag and ID
    merged_df = pd.merge(merged_df, education_2022[['ID', '계열화']], left_on='knowledgeTag', right_on='ID', how='left')

    # Drop the redundant 'ID' column after the merge
    merged_df = merged_df.drop(columns=['ID'], errors='ignore')

    # Ensure columns are numeric (especially answerCode, difficultyLevel, etc.)
    merged_df['answerCode'] = pd.to_numeric(merged_df['answerCode'], errors='coerce')
    merged_df['difficultyLevel'] = pd.to_numeric(merged_df['difficultyLevel'], errors='coerce')
    merged_df['discriminationLevel'] = pd.to_numeric(merged_df['discriminationLevel'], errors='coerce')
    merged_df['theta'] = pd.to_numeric(merged_df['theta'], errors='coerce')
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')


    return merged_df

# # PostgreSQL 연결 정보 설정
# db_config = {
#     'user': 'postgres',
#     'password': 'qwe123',
#     'host': '127.0.0.1',
#     'port': '5432',
#     'database': 'project'
# }

# # SQLAlchemy 엔진 생성
# engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# # 테이블 생성 SQL 쿼리
# create_prerequisite_table = """
# CREATE TABLE IF NOT EXISTS prerequisite_recommendations (
#     learner_id VARCHAR(50),
#     knowledgeTag INTEGER,
#     선수학습_Chapter_Name VARCHAR(255),
#     계열화 VARCHAR(255),
#     영역 VARCHAR(255)
# );
# """

# create_successor_table = """
# CREATE TABLE IF NOT EXISTS successor_recommendations (
#     learner_id VARCHAR(50),
#     knowledgeTag INTEGER,
#     후속학습_Chapter_Name VARCHAR(255),
#     계열화 VARCHAR(255),
#     영역 VARCHAR(255)
# );
# """

# def create_table_if_not_exists(table_name, create_table_query):
#     """테이블이 없으면 생성하는 함수"""
#     with engine.connect() as conn:
#         # 테이블 존재 여부 확인 쿼리
#         check_table_exists_query = f"""
#         SELECT EXISTS (
#             SELECT FROM information_schema.tables 
#             WHERE table_name = '{table_name}'
#         );
#         """
#         result = conn.execute(text(check_table_exists_query)).scalar()
#         # 테이블이 없으면 생성
#         if not result:
#             conn.execute(text(create_table_query))
#             print(f"테이블 {table_name}이 생성되었습니다.")
#         else:
#             print(f"테이블 {table_name}이 이미 존재합니다.")

# def insert_prerequisite_data(data):
#     """선수학습 추천 데이터를 적재"""
#     # 테이블 생성 확인 및 생성
#     create_table_if_not_exists('prerequisite_recommendations', create_prerequisite_table)

#     # 데이터 삽입
#     insert_query = """
#     INSERT INTO prerequisite_recommendations (learner_id, knowledgeTag, 선수학습_Chapter_Name, 계열화, 영역)
#     VALUES (:learner_id, :knowledgeTag, :선수학습_Chapter_Name, :계열화, :영역);
#     """
#     with engine.connect() as conn:
#         conn.execute(text(insert_query), data)

# def insert_successor_data(data):
#     """후속학습 추천 데이터를 적재"""
#     # 테이블 생성 확인 및 생성
#     create_table_if_not_exists('successor_recommendations', create_successor_table)

#     # 데이터 삽입
#     insert_query = """
#     INSERT INTO successor_recommendations (learner_id, knowledgeTag, 후속학습_Chapter_Name, 계열화, 영역)
#     VALUES (:learner_id, :knowledgeTag, :후속학습_Chapter_Name, :계열화, :영역);
#     """
#     with engine.connect() as conn:
#         conn.execute(text(insert_query), data)


# PostgreSQL 연결 정보 설정
db_config = {
    'user': 'postgres',
    'password': 'qwe123',
    'host': '10.41.2.78',
    'port': '5432',
    'database': 'project'
}

# SQLAlchemy 엔진 생성
engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# 테이블 생성 SQL 쿼리
create_prerequisite_table = """
CREATE TABLE IF NOT EXISTS prerequisite_recommendations_save (
    learner_id VARCHAR(50),
    knowledgeTag INTEGER,
    선수학습_Chapter_Name VARCHAR(255),
    계열화 VARCHAR(255),
    영역 VARCHAR(255),
    CONSTRAINT unique_prerequisite UNIQUE (learner_id, knowledgeTag)
);
"""

create_successor_table = """
CREATE TABLE IF NOT EXISTS successor_recommendations_save (
    learner_id VARCHAR(50),
    knowledgeTag INTEGER,
    후속학습_Chapter_Name VARCHAR(255),
    계열화 VARCHAR(255),
    영역 VARCHAR(255),
    CONSTRAINT unique_successor UNIQUE (learner_id, knowledgeTag)
);
"""

def create_table_if_not_exists(table_name, create_table_query):
    """테이블이 없으면 생성하는 함수"""
    with engine.connect() as conn:
        try:
            conn.execute(text(create_table_query))
            conn.execute(text("COMMIT;"))  # 커밋 명시적으로 수행
            print(f"테이블 {table_name}이 생성되었습니다.")
        except Exception as e:
            print(f"테이블 생성 중 오류가 발생했습니다: {e}")

def insert_prerequisite_data(data):
    """선수학습 추천 데이터를 적재"""
    # 테이블 생성 확인 및 생성
    create_table_if_not_exists('prerequisite_recommendations_save', create_prerequisite_table)

    # 데이터 삽입
    insert_query = """
    INSERT INTO prerequisite_recommendations_save (learner_id, knowledgeTag, 선수학습_Chapter_Name, 계열화, 영역)
    VALUES (:learner_id, :knowledgeTag, :선수학습_Chapter_Name, :계열화, :영역)
    ON CONFLICT (learner_id, knowledgeTag) DO NOTHING;
    """
    with engine.connect() as conn:
        try:
            with conn.begin() as transaction:
                result = conn.execute(text(insert_query), data)
                transaction.commit()  # 커밋 명시적으로 수행
                if result.rowcount == 0:
                    print(f"중복 데이터가 발견되어 저장되지 않았습니다: {data}")
        except IntegrityError as e:
            print(f"중복 데이터로 인해 저장되지 않았습니다: {e}")
        except Exception as e:
            print(f"데이터 삽입 중 오류가 발생했습니다: {e}")

def insert_successor_data(data):
    """후속학습 추천 데이터를 적재"""
    # 테이블 생성 확인 및 생성
    create_table_if_not_exists('successor_recommendations_save', create_successor_table)

    # 데이터 삽입
    insert_query = """
    INSERT INTO successor_recommendations_save (learner_id, knowledgeTag, 후속학습_Chapter_Name, 계열화, 영역)
    VALUES (:learner_id, :knowledgeTag, :후속학습_Chapter_Name, :계열화, :영역)
    ON CONFLICT (learner_id, knowledgeTag) DO NOTHING;
    """
    with engine.connect() as conn:
        try:
            with conn.begin() as transaction:
                result = conn.execute(text(insert_query), data)
                transaction.commit()  # 커밋 명시적으로 수행
                if result.rowcount == 0:
                    print(f"중복 데이터가 발견되어 저장되지 않았습니다: {data}")
        except IntegrityError as e:
            print(f"중복 데이터로 인해 저장되지 않았습니다: {e}")
        except Exception as e:
            print(f"데이터 삽입 중 오류가 발생했습니다: {e}")

def check_table_exists(table_name):
    """테이블 존재 여부를 확인하는 함수"""
    with engine.connect() as conn:
        check_table_exists_query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        );
        """
        result = conn.execute(text(check_table_exists_query)).scalar()
        return result