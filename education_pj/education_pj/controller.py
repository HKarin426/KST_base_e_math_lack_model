import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import re
from collections import defaultdict
from scipy.spatial.distance import mahalanobis



# # 데이터 로드
# merged_df, label_math_ele = load_data()



class EnhancedBinaryNN(nn.Module):
    def __init__(self):
        super(EnhancedBinaryNN, self).__init__()
        # 더 깊고 복잡한 네트워크
        self.fc1 = nn.Linear(4, 256)  # 첫 번째 은닉층의 유닛 수를 256으로 증가
        self.fc2 = nn.Linear(256, 128)  # 두 번째 은닉층의 유닛 수를 128로 설정
        self.fc3 = nn.Linear(128, 64)   # 세 번째 은닉층 추가
        self.fc4 = nn.Linear(64, 32)    # 네 번째 은닉층 추가
        self.fc5 = nn.Linear(32, 1)     # 출력층
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)  # 드롭아웃 확률을 40%로 증가
        self.batch_norm1 = nn.BatchNorm1d(256)  # 배치 정규화 추가
        self.batch_norm2 = nn.BatchNorm1d(128)  # 배치 정규화 추가
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(self.relu(self.batch_norm2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
        
# Load the model
def load_model(model_path):
    model = EnhancedBinaryNN()
    model.load_state_dict(torch.load('./enhanced_binary_nn_model.pth'))
    model.eval()
    print("Model loaded from 'enhanced_binary_nn_model.pth'")
    return model


# Utility functions
def extract_semester(semester_str):
    if pd.isna(semester_str):
        return ''
    match = re.search(r'(\d+)-(\d+)', semester_str)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

def parse_series_order(order_str):
    if not order_str:
        return ('ZZZ', float('inf'), float('inf'))
    match = re.match(r'([A-Z]+)-(\d+)-(\d+)', order_str)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)))
    return ('ZZZ', float('inf'), float('inf'))

def get_unique_id_name_pairs(df):
    to_id_name = dict(zip(df['to_id'], df['to_chapter_name']))
    from_id_name = dict(zip(df['from_id'], df['from_chapter_name']))
    all_ids = set(to_id_name.keys()).union(set(from_id_name.keys()))
    unique_id_name_pairs = {}
    unique_id_semesters = {}
    for id_ in all_ids:
        to_name = to_id_name.get(id_, '')
        from_name = from_id_name.get(id_, '')
        if to_name and from_name:
            unique_id_name_pairs[id_] = f'{to_name} / {from_name}'
        elif to_name:
            unique_id_name_pairs[id_] = to_name
        elif from_name:
            unique_id_name_pairs[id_] = from_name
        from_semester = df[df['from_id'] == id_]['from_semester'].values
        to_semester = df[df['to_id'] == id_]['to_semester'].values
        if len(from_semester) > 0:
            unique_id_semesters[id_] = extract_semester(from_semester[0])
        elif len(to_semester) > 0:
            unique_id_semesters[id_] = extract_semester(to_semester[0])
        else:
            unique_id_semesters[id_] = (None, None)
    return unique_id_name_pairs, unique_id_semesters

# controller.py 수정: 전역에서 education_2022를 사용하지 않고 함수 인자로 처리
def process_label_math_data(label_math_ele, education_2022):
    # 본개념과 ID-이름 쌍, 학기 정보 가져오기
    unique_id_name_pairs, unique_id_semesters = get_unique_id_name_pairs(label_math_ele)

    # education_2022 데이터를 ID를 기준으로 매칭할 수 있도록 딕셔너리 생성
    education_mapping = education_2022.set_index('ID').to_dict('index')
    
    predecessors = defaultdict(list)
    successors = defaultdict(list)  # successors가 여기서 초기화됨

    to_chapter_names = dict(zip(label_math_ele['to_id'], label_math_ele['to_chapter_name']))
    from_chapter_names = dict(zip(label_math_ele['from_id'], label_math_ele['from_chapter_name']))

    for _, row in label_math_ele.iterrows():
        from_id = row['from_id']
        to_id = row['to_id']
        predecessors[from_id].append(to_id)
        successors[to_id].append(from_id)  # successors에 값이 추가됨

    return unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping



def analyze_student_performance(learner_id, df):
    student_df = df[df['learnerID'] == learner_id].copy()
    if student_df.empty:
        print(f"학습자 {learner_id}의 데이터가 없습니다.")
        return None, None
    if student_df['answerCode'].dtype == 'object':
        student_df['answerCode'] = pd.to_numeric(student_df['answerCode'], errors='coerce')
    correct_df = student_df[student_df['answerCode'] == 1]
    incorrect_df = student_df[student_df['answerCode'] == 0]
    return correct_df['knowledgeTag'].tolist(), incorrect_df['knowledgeTag'].tolist()

def get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors):
    main_concept = {
        '본개념': node_id,
        '본개념_Chapter_Name': unique_id_name_pairs.get(node_id, '정보 없음'),
        '학년-학기': unique_id_semesters.get(node_id, (None, None))  # unique_id_semesters 추가 확인
    }
    main_concept['선수학습'] = sorted(
        predecessors.get(node_id, []),
        key=lambda x: parse_series_order(education_mapping.get(x, {}).get('계열화', '')),
        reverse=True
    )
    main_concept['후속학습'] = sorted(
        successors.get(node_id, []),
        key=lambda x: parse_series_order(education_mapping.get(x, {}).get('계열화', ''))
    )
    return main_concept


def exclude_correct_predecessors_from_incorrect(correct_preds, incorrect_preds):
    correct_set = set(id_name for preds in correct_preds.values() for id_name in preds)
    incorrect_set = set(id_name for preds in incorrect_preds.values() for id_name in preds)
    return incorrect_set - correct_set

def get_chapter_serialization_info(predecessor_ids, unique_id_name_pairs, education_mapping):
    serialization_info = []
    for id_ in predecessor_ids:
        chapter_name = unique_id_name_pairs.get(id_, '정보 없음')
        info = education_mapping.get(id_, {})
        serialization = info.get('계열화', '정보 없음')
        serialization_info.append({
            '선수학습_ID': id_,
            '선수학습_Chapter_Name': chapter_name,
            '계열화': serialization
        })
    return pd.DataFrame(serialization_info)

def extract_first_part(df):
    df['첫 번째 부분'] = df['계열화'].apply(lambda x: x.split('>')[0].strip())
    return df[['knowledgeTag', '첫 번째 부분']]

def filter_successor_recommendations(successor_df, prerequisite_areas):
    prerequisite_parts = prerequisite_areas['첫 번째 부분'].unique()
    filtered_successors = successor_df[~successor_df['계열화'].apply(lambda x: x.split('>')[0].strip()).isin(prerequisite_parts)]
    return filtered_successors

def extract_prefix(serialization_code):
    # Extract prefix from serialization code (before the first dash or space)
    return serialization_code.split('-')[0] if '-' in serialization_code else serialization_code.split(' ')[0]

# recommend_concept 함수에서 교집합 없는 경우 처리 로직 수정
def recommend_concept(learner_id, df, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors):
    correct_knowledge_tags, incorrect_knowledge_tags = analyze_student_performance(learner_id, df)
    if correct_knowledge_tags is None or incorrect_knowledge_tags is None:
        return pd.DataFrame(), pd.DataFrame()

    correct_predecessors = {tag: get_concepts(tag, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors)['선수학습'] for tag in correct_knowledge_tags}
    incorrect_predecessors = {tag: get_concepts(tag, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors)['선수학습'] for tag in incorrect_knowledge_tags}

    unique_incorrect_only_preds = exclude_correct_predecessors_from_incorrect(correct_predecessors, incorrect_predecessors)

    # 교집합 없는 경우 처리
    if not unique_incorrect_only_preds:
        print("교집합이 없는 학생입니다. 선행학습만 추천합니다.")
        successor_recommendations_df = recommend_successors_based_on_previous_learning(
            incorrect_knowledge_tags, unique_id_name_pairs, education_mapping, predecessors, successors
        )
        return pd.DataFrame(), successor_recommendations_df

    # 기존 로직 유지
    serialization_info_df = get_chapter_serialization_info(unique_incorrect_only_preds, unique_id_name_pairs, education_mapping)
    
    # Prepare student data for model input
    student_df = df[df['learnerID'] == learner_id].copy() 
    if 'weighted_score' not in student_df.columns:
        student_df.loc[:, 'weighted_score'] = 1
    X_student = student_df[['difficultyLevel', 'discriminationLevel', 'theta', 'guessLevel']].values
    X_student_tensor = torch.tensor(X_student, dtype=torch.float32)
    
    # Make predictions using the model
    model.eval()
    with torch.no_grad():
        model_output = model(X_student_tensor).squeeze().numpy()
    
    # Check if model_output is not empty
    if model_output.size == 0:
        print("Error: Model output is empty.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Add prediction probabilities to student_df
    student_df.loc[:, 'predicted_probability'] = model_output

    # Check if 'predicted_probability' column is created
    if 'predicted_probability' not in student_df.columns:
        print("Error: 'predicted_probability' column is missing in student_df.")
        return pd.DataFrame(), pd.DataFrame()

    # Prepare recommendations based on predicted probabilities
    recommendations = []
    for _, row in serialization_info_df.iterrows():
        concept_id = row['선수학습_ID']
        if concept_id in student_df['knowledgeTag'].values:
            prob = student_df[student_df['knowledgeTag'] == concept_id]['predicted_probability'].values
            probability = prob[0] if prob.size > 0 else 0.0
            recommendations.append({
                'knowledgeTag': concept_id,
                'predicted_probability': probability,
                '선수학습_Chapter_Name': row['선수학습_Chapter_Name'],
                '계열화': row['계열화']
            })
    
    # Convert recommendations to DataFrame
    recommendations_df = pd.DataFrame(recommendations)

    # Handle the case when recommendations_df is empty
    if recommendations_df.empty:
        print("추천 결과가 없습니다.")
        # 수정 코드: 디버깅 메시지 추가
        # print("incorrect_knowledge_tags:", incorrect_knowledge_tags)
        # print("unique_id_name_pairs:", unique_id_name_pairs)
        # print("education_mapping:", education_mapping)
        # print("predecessors:", predecessors)
        # print("successors:", successors)
        successor_recommendations_df = recommend_successors_based_on_previous_learning(
            incorrect_knowledge_tags, unique_id_name_pairs, education_mapping, predecessors, successors  # 'successors' 추가
        )
        return recommendations_df, successor_recommendations_df

    # Check if 'predicted_probability' column is present in recommendations_df
    if 'predicted_probability' not in recommendations_df.columns:
        print("Error: 'predicted_probability' column is missing in recommendations_df.")
    
    recommendations_df = recommendations_df.sort_values(by='predicted_probability', ascending=False)
    
    # Extract successor recommendations (기존 로직 유지)
    successor_recommendations = []
    for tag in incorrect_knowledge_tags:
        concepts = get_concepts(tag, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors)
        for succ in concepts['후속학습']:
            successor_area = extract_prefix(education_mapping.get(succ, {}).get('계열화', '정보 없음'))
            if successor_area not in [extract_prefix(x) for x in serialization_info_df['계열화']]:
                successor_recommendations.append({
                    'knowledgeTag': succ,
                    'predicted_probability': 0.0,
                    '후속학습_Chapter_Name': unique_id_name_pairs.get(succ, '정보 없음'),
                    '계열화': education_mapping.get(succ, {}).get('계열화', '정보 없음')
                })
    
    # Convert successor recommendations to DataFrame
    successor_recommendations_df = pd.DataFrame(successor_recommendations)
    
    # Calculate Mahalanobis distance for successor recommendations
    if not successor_recommendations_df.empty:
        correct_answers = df[df['answerCode'] == 1]
        mean_correct = correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
        cov_correct = np.cov(correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']], rowvar=False)
        
        distances = {}
        for _, row in successor_recommendations_df.iterrows():
            knowledge_tag = row['knowledgeTag']
            if knowledge_tag in df['knowledgeTag'].values:
                row_data = df[df['knowledgeTag'] == knowledge_tag][['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
                distance = mahalanobis(row_data, mean_correct, np.linalg.inv(cov_correct))
                distances[knowledge_tag] = distance
        
        successor_recommendations_df['Mahalanobis_distance'] = successor_recommendations_df['knowledgeTag'].apply(lambda x: distances.get(x, float('inf')))
        successor_recommendations_df = successor_recommendations_df.sort_values(by='Mahalanobis_distance', ascending=True)

    return recommendations_df, successor_recommendations_df


# 수정된 recommend_successors_based_on_previous_learning 함수
def recommend_successors_based_on_previous_learning(incorrect_knowledge_tags, unique_id_name_pairs, education_mapping, predecessors, successors):
    # Use the existing function for successor recommendations if necessary
    successor_recommendations = []
    for tag in incorrect_knowledge_tags:
        concepts = get_concepts(tag, unique_id_name_pairs, {}, education_mapping, predecessors, successors)
        for succ in concepts['후속학습']:
            successor_area = extract_prefix(education_mapping.get(succ, {}).get('계열화', '정보 없음'))
            successor_recommendations.append({
                'knowledgeTag': succ,
                'predicted_probability': 0.0,
                '후속학습_Chapter_Name': unique_id_name_pairs.get(succ, '정보 없음'),
                '계열화': education_mapping.get(succ, {}).get('계열화', '정보 없음')
            })

    successor_recommendations_df = pd.DataFrame(successor_recommendations)
    return successor_recommendations_df


# def recommend_successors_based_on_previous_learning(incorrect_knowledge_tags, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors):
#     # Use the existing function for successor recommendations if necessary
#     successor_recommendations = []
#     for tag in incorrect_knowledge_tags:
#         # 여기서 unique_id_semesters를 추가로 전달합니다.
#         concepts = get_concepts(tag, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors)
#         for succ in concepts['후속학습']:
#             successor_area = extract_prefix(education_mapping.get(succ, {}).get('계열화', '정보 없음'))
#             successor_recommendations.append({
#                 'knowledgeTag': succ,
#                 'predicted_probability': 0.0,
#                 '후속학습_Chapter_Name': unique_id_name_pairs.get(succ, '정보 없음'),
#                 '계열화': education_mapping.get(succ, {}).get('계열화', '정보 없음')
#             })

    
#     # Convert successor recommendations to DataFrame
#     successor_recommendations_df = pd.DataFrame(successor_recommendations)
    
#     # Calculate Mahalanobis distance for successor recommendations
#     if not successor_recommendations_df.empty:
#         correct_answers = merged_df[merged_df['answerCode'] == 1]
#         mean_correct = correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
#         cov_correct = np.cov(correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']], rowvar=False)
        
#         distances = {}
#         for _, row in successor_recommendations_df.iterrows():
#             knowledge_tag = row['knowledgeTag']
#             if knowledge_tag in merged_df['knowledgeTag'].values:
#                 row_data = merged_df[merged_df['knowledgeTag'] == knowledge_tag][['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
#                 distance = mahalanobis(row_data, mean_correct, np.linalg.inv(cov_correct))
#                 distances[knowledge_tag] = distance
        
#         successor_recommendations_df['Mahalanobis_distance'] = successor_recommendations_df['knowledgeTag'].apply(lambda x: distances.get(x, float('inf')))
#         successor_recommendations_df = successor_recommendations_df.sort_values(by='Mahalanobis_distance', ascending=True)

#     return successor_recommendations_df




def recommend_all(learner_id, df, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors):
    # Get recommendations for prerequisites
    prerequisite_recommendations, successor_recommendations = recommend_concept(
        learner_id, df, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors
    )
    
    # Check if prerequisite recommendations are empty and handle accordingly
    if prerequisite_recommendations.empty:
        print("선수학습 개념이 없으므로 후속 학습 개념을 추천합니다.")
        return pd.DataFrame(), successor_recommendations

    return prerequisite_recommendations, successor_recommendations


def map_area(serialization_code):
    # Define the mapping from serialization code to area
    area_mapping = {
        'A': '수와 연산',
        'B': '변화와 관계',
        'C': '도형',
        'D': '측정',
        'E': '자료와 가능성'
    }
    
    # Extract the prefix from serialization code
    prefix = extract_prefix(serialization_code)
    
    # Map the prefix to area
    return area_mapping.get(prefix, '기타')

def recommend_one_per_area(recommendations_df):
    # Check if 'predicted_probability' column is present
    if 'predicted_probability' not in recommendations_df.columns:
        print("Error: 'predicted_probability' column is missing in recommendations_df.")
        print("선수학습 개념이 없습니다.")  # Print a message in case no recommendations are available
        return pd.DataFrame()  # Return an empty DataFrame

    # Filter out rows where '계열화' is '정보 없음'
    recommendations_df = recommendations_df[recommendations_df['계열화'] != '정보 없음']
    
    # Add '영역' column based on '계열화'
    recommendations_df['영역'] = recommendations_df['계열화'].apply(map_area)
    
    # Group by '영역' and select the highest predicted probability in each group
    top_recommendations = recommendations_df.loc[
        recommendations_df.groupby('영역')['predicted_probability'].idxmax()
    ]
    
    # Sort by predicted probability in descending order
    top_recommendations = top_recommendations.sort_values(by='predicted_probability', ascending=False)
    
    # Remove unnecessary columns
    top_recommendations = top_recommendations.drop(columns=['predicted_probability'])
    
    return top_recommendations


# Mahalanobis 거리 계산 함수 수정 - merged_df를 인자로 전달
def calculate_mahalanobis_distance(successor_recommendations_df, merged_df):
    correct_answers = merged_df[merged_df['answerCode'] == 1]
    mean_correct = correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
    cov_correct = np.cov(correct_answers[['difficultyLevel', 'discriminationLevel', 'guessLevel']], rowvar=False)

    distances = {}
    for _, row in successor_recommendations_df.iterrows():
        knowledge_tag = row['knowledgeTag']
        if knowledge_tag in merged_df['knowledgeTag'].values:
            row_data = merged_df[merged_df['knowledgeTag'] == knowledge_tag][['difficultyLevel', 'discriminationLevel', 'guessLevel']].mean().values
            distance = mahalanobis(row_data, mean_correct, np.linalg.inv(cov_correct))
            distances[knowledge_tag] = distance

    successor_recommendations_df['Mahalanobis_distance'] = successor_recommendations_df['knowledgeTag'].apply(lambda x: distances.get(x, float('inf')))
    return successor_recommendations_df

# 추천 후계 학습을 선택하는 함수 수정 - .copy()로 슬라이스 문제 해결
def recommend_one_per_area_successors(successor_recommendations_df, merged_df):
    # '계열화'가 '정보 없음'인 항목 필터링 및 슬라이스 복사
    successor_recommendations_df = successor_recommendations_df[successor_recommendations_df['계열화'] != '정보 없음'].copy()

    # '영역' 컬럼을 추가 (.loc[] 사용, SettingWithCopyWarning 해결)
    successor_recommendations_df.loc[:, '영역'] = successor_recommendations_df['계열화'].apply(map_area)

    # 'Mahalanobis_distance' 컬럼이 존재하는지 확인 후 계산
    if 'Mahalanobis_distance' not in successor_recommendations_df.columns:
        print("Error: 'Mahalanobis_distance' column is missing. Calculating distances...")
        successor_recommendations_df = calculate_mahalanobis_distance(successor_recommendations_df, merged_df)

    # 그룹별 가장 작은 Mahalanobis 거리를 가진 항목 선택
    top_successor_recommendations = successor_recommendations_df.loc[
        successor_recommendations_df.groupby('영역')['Mahalanobis_distance'].idxmin()
    ]

    # 필요 없는 컬럼 제거 후 반환
    top_successor_recommendations = top_successor_recommendations.drop(columns=['predicted_probability', 'Mahalanobis_distance'])
    
    return top_successor_recommendations
