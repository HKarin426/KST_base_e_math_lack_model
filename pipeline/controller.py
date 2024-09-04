# concept_analysis.py

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# 데이터 전처리
def preprocess_data(label_math_ele):
    label_math_ele['from_chapter_name'].fillna('', inplace=True)
    label_math_ele['to_chapter_name'].fillna('', inplace=True)
    return label_math_ele

# 학기 추출
def extract_semester(semester_str):
    if pd.isna(semester_str):
        return ''
    match = re.search(r'(\d+)-(\d+)', semester_str)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

# 시리즈 순서 파싱
def parse_series_order(order_str):
    if not order_str:
        return ('ZZZ', float('inf'), float('inf'))
    match = re.match(r'([A-Z]+)-(\d+)-(\d+)', order_str)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)))
    return ('ZZZ', float('inf'), float('inf'))

# 고유 ID와 이름 쌍 가져오기
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

# 교육 매핑 생성
def create_education_mapping(education_2022):
    return education_2022.set_index('ID').to_dict('index')

# 선수학습 및 후속학습 생성
def create_predecessors_successors(label_math_ele):
    predecessors = defaultdict(list)
    successors = defaultdict(list)
    for _, row in label_math_ele.iterrows():
        from_id = row['from_id']
        to_id = row['to_id']
        predecessors[from_id].append(to_id)
        successors[to_id].append(from_id)
    return predecessors, successors

# 학습자 성과 분석
def analyze_student_performance(learner_id, df):
    student_df = df[df['learnerID'] == learner_id].copy()
    if student_df.empty:
        print(f"학습자 {learner_id}의 데이터가 없습니다.")
        return None, None
    if student_df['answerCode'].dtype == 'object':
        student_df.loc[:, 'answerCode'] = pd.to_numeric(student_df['answerCode'], errors='coerce')
    correct_df = student_df[student_df['answerCode'] == 1]
    incorrect_df = student_df[student_df['answerCode'] == 0]
    correct_knowledge_tags = correct_df['knowledgeTag'].dropna()
    incorrect_knowledge_tags = incorrect_df['knowledgeTag'].dropna()
    return correct_knowledge_tags.tolist(), incorrect_knowledge_tags.tolist()

# 가장 빈도수가 높은 태그 추출
def get_most_common_tag(tags_list):
    all_tags = [tag for sublist in tags_list for tag in sublist]
    if all_tags:
        return pd.Series(all_tags).value_counts().idxmax()
    return None

# 개념 정보 가져오기
def get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping):
    main_concept = {
        '본개념': node_id,
        '본개념_Chapter_Name': unique_id_name_pairs.get(node_id, '정보 없음'),
        '학년-학기': unique_id_semesters.get(node_id, (None, None))
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

# 데이터프레임을 이용하여 개념 관련 데이터 추출
def get_concepts_df(node_id):
    # 여기에 실제로 데이터프레임을 불러오고 필터링하여 반환하는 로직을 구현해야 합니다.
    # 예시로 빈 데이터프레임을 반환합니다.
    # 실제 데이터 로딩과 필터링 로직을 구현하세요.
    concept_df = pd.DataFrame({'Concept': [node_id], 'Detail': ['Sample detail']})
    predecessors_df = pd.DataFrame({'Predecessor': [node_id], 'Detail': ['Sample predecessor detail']})
    successors_df = pd.DataFrame({'Successor': [node_id], 'Detail': ['Sample successor detail']})
    return concept_df, predecessors_df, successors_df

# 데이터프레임에 교육 정보 추가
def add_education_info(df, id_col, education_mapping):
    df['영역명'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('영역명', '정보 없음'))
    df['내용요소'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('내용요소', '정보 없음'))
    df['계열화'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('계열화', '정보 없음'))
    return df

# NaN 값을 문자열로 교체
def replace_nan_with_string(df, replace_str='정보 없음'):
    return df.replace({np.nan: replace_str})

# 학기 필터링된 개념 정보
def get_concepts_with_filters(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping, semester=None):
    main_concept = get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping)
    if semester:
        main_concept['선수학습'] = [concept for concept in main_concept['선수학습']
                             if unique_id_semesters.get(concept, (None, None))[0] <= semester]
        main_concept['후속학습'] = [concept for concept in main_concept['후속학습']
                             if unique_id_semesters.get(concept, (None, None))[0] <= semester]
    return main_concept


'''

import pandas as pd
import numpy as np
import re
from collections import defaultdict

def preprocess_data(label_math_ele):
    label_math_ele['from_chapter_name'].fillna('', inplace=True)
    label_math_ele['to_chapter_name'].fillna('', inplace=True)
    return label_math_ele

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

def create_education_mapping(education_2022):
    return education_2022.set_index('ID').to_dict('index')

def create_predecessors_successors(label_math_ele):
    predecessors = defaultdict(list)
    successors = defaultdict(list)
    for _, row in label_math_ele.iterrows():
        from_id = row['from_id']
        to_id = row['to_id']
        predecessors[from_id].append(to_id)
        successors[to_id].append(from_id)
    return predecessors, successors

def analyze_student_performance(learner_id, df):
    student_df = df[df['learnerID'] == learner_id].copy()
    if student_df.empty:
        print(f"학습자 {learner_id}의 데이터가 없습니다.")
        return None, None
    if student_df['answerCode'].dtype == 'object':
        student_df.loc[:, 'answerCode'] = pd.to_numeric(student_df['answerCode'], errors='coerce')
    correct_df = student_df[student_df['answerCode'] == 1]
    incorrect_df = student_df[student_df['answerCode'] == 0]
    correct_knowledge_tags = correct_df['knowledgeTag'].dropna()
    incorrect_knowledge_tags = incorrect_df['knowledgeTag'].dropna()
    return correct_knowledge_tags.tolist(), incorrect_knowledge_tags.tolist()

def get_most_common_tag(tags_list):
    all_tags = [tag for sublist in tags_list for tag in sublist]
    if all_tags:
        return pd.Series(all_tags).value_counts().idxmax()
    return None

def get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping):
    main_concept = {
        '본개념': node_id,
        '본개념_Chapter_Name': unique_id_name_pairs.get(node_id, '정보 없음'),
        '학년-학기': unique_id_semesters.get(node_id, (None, None))
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

def add_education_info(df, id_col, education_mapping):
    df['영역명'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('영역명', '정보 없음'))
    df['내용요소'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('내용요소', '정보 없음'))
    df['계열화'] = df[id_col].apply(lambda x: education_mapping.get(x, {}).get('계열화', '정보 없음'))
    return df

def replace_nan_with_string(df, replace_str='정보 없음'):
    return df.replace({np.nan: replace_str})

def get_concepts_with_filters(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping, semester=None):
    main_concept = get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping)
    if semester:
        main_concept['선수학습'] = [concept for concept in main_concept['선수학습']
                             if unique_id_semesters.get(concept, (None, None))[0] <= semester]
        main_concept['후속학습'] = [concept for concept in main_concept['후속학습']
                             if unique_id_semesters.get(concept, (None, None))[0] <= semester]
    return main_concept
'''