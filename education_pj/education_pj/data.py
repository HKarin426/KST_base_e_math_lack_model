import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine

# DB Configuration
def load_data():
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

    # MongoDB connection
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

    # Fetch data from collections
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

    return merged_df, label_math_ele

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
    merged_df, label_math_ele = load_data()
    education_2022 = load_education_data()

    # Ensure ID column in education_2022 is numeric to match with merged_df's knowledgeTag
    education_2022['ID'] = pd.to_numeric(education_2022['ID'], errors='coerce', downcast='integer')

    # Remove leading/trailing whitespace if necessary (in case of string issues)
    education_2022['ID'] = education_2022['ID'].astype(str).str.strip()
    merged_df['knowledgeTag'] = merged_df['knowledgeTag'].astype(str).str.strip()

    # Convert to numeric again if needed
    education_2022['ID'] = pd.to_numeric(education_2022['ID'], errors='coerce', downcast='integer')
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    # Check if there are common values between merged_df['knowledgeTag'] and education_2022['ID']
    common_ids = pd.Series(merged_df['knowledgeTag']).isin(education_2022['ID'])
    print(f"Common values between knowledgeTag and ID: {common_ids.sum()}")  # 공통된 ID의 수를 출력

    if common_ids.sum() == 0:
        print("No common values between knowledgeTag and ID. Check the source data.")
        return merged_df  # 공통된 값이 없다면 함수 종료

    # Merge education_2022 with merged_df on knowledgeTag and ID
    merged_df = pd.merge(merged_df, education_2022[['ID', '계열화']], left_on='knowledgeTag', right_on='ID', how='left')

    # Drop the redundant 'ID' column after the merge
    merged_df = merged_df.drop(columns=['ID'], errors='ignore')

    # Ensure columns are numeric (especially answerCode)
    merged_df['answerCode'] = pd.to_numeric(merged_df['answerCode'], errors='coerce')  # Converts non-numeric values to NaN
    merged_df['difficultyLevel'] = pd.to_numeric(merged_df['difficultyLevel'], errors='coerce')
    merged_df['discriminationLevel'] = pd.to_numeric(merged_df['discriminationLevel'], errors='coerce')
    merged_df['theta'] = pd.to_numeric(merged_df['theta'], errors='coerce')
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    # Print the merged DataFrame to check for the '계열화' column
    print("Merged DataFrame head: ")
    print(merged_df[['learnerID', 'knowledgeTag', '계열화']].head())  # Preview relevant columns

    return merged_df



