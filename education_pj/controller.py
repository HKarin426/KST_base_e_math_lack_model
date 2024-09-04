import pandas as pd
from collections import defaultdict
import re
import numpy as np
from data import fetch_mongo_data, fetch_postgres_data, fetch_label_math

def get_merged_data():
    merged_df = fetch_mongo_data()
    education_2022 = fetch_postgres_data()
    label_math = fetch_label_math()

    label_math['from_g'] = label_math['from_semester'].apply(lambda x: x.split('-')[0])
    label_math['to_g'] = label_math['to_semester'].apply(lambda x: x.split('-')[0])

    label_math_ele = label_math[
        ~(
            ((label_math['from_g'] == '고등') | (label_math['to_g'] == '고등')) |
            ((label_math['from_g'] == '중등') & (label_math['to_g'] == '중등'))
        )
    ]

    replacement_dict = {
        '중등-중1-1학기': '중등-중7-1학기',
        '중등-중2-2학기': '중등-중8-2학기',
        '중등-중1-2학기': '중등-중7-2학기',
        '중등-중2-1학기': '중등-중8-1학기',
        '중등-중3-1학기': '중등-중9-1학기',
        '중등-중3-2학기': '중등-중9-2학기'
    }

    label_math_ele.loc[label_math_ele['from_g'] == '중등', 'from_semester'] = label_math_ele['from_semester'].replace(replacement_dict)

    label_math_ele['from_chapter_name'].fillna('', inplace=True)
    label_math_ele['to_chapter_name'].fillna('', inplace=True)

    return merged_df, education_2022, label_math_ele

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

def get_concepts(node_id, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters):
    main_concept = {
        '본개념': node_id,
        '본개념_Chapter_Name': unique_id_name_pairs.get(node_id, '정보 없음'),
        '학년-학기': unique_id_semesters.get(node_id, (None, None))
    }
    main_concept['선수학습'] = sorted(
        predecessors.get(node_id, []),
        key=lambda x: parse_series_order(education_mapping.get(x, {}).get('계열화', ''))
    , reverse=True)
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

def get_concepts_df(node_id, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters):
    concepts = get_concepts(node_id, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters)
    main_concept_df = pd.DataFrame([{
        '본개념_ID': concepts['본개념'],
        '본개념_Chapter_Name': concepts['본개념_Chapter_Name'],
        '학년-학기': f"{concepts['학년-학기'][0]}-{concepts['학년-학기'][1]}"
    }])
    main_concept_df = add_education_info(main_concept_df, '본개념_ID', education_mapping)
    main_concept_df = replace_nan_with_string(main_concept_df)
    predecessors_df = pd.DataFrame({
        '선수학습_ID': concepts['선수학습'],
        '선수학습_Chapter_Name': [unique_id_name_pairs.get(id, '정보 없음') for id in concepts['선수학습']],
        '학년-학기': [f"{unique_id_semesters.get(id, (None, None))[0]}-{unique_id_semesters.get(id, (None, None))[1]}" for id in concepts['선수학습']]
    })
    predecessors_df = add_education_info(predecessors_df, '선수학습_ID', education_mapping)
    predecessors_df = replace_nan_with_string(predecessors_df)
    successors_df = pd.DataFrame({
        '후속학습_ID': concepts['후속학습'],
        '후속학습_Chapter_Name': [unique_id_name_pairs.get(id, '정보 없음') for id in concepts['후속학습']],
        '학년-학기': [f"{unique_id_semesters.get(id, (None, None))[0]}-{unique_id_semesters.get(id, (None, None))[1]}" for id in concepts['후속학습']]
    })
    successors_df = add_education_info(successors_df, '후속학습_ID', education_mapping)
    successors_df = replace_nan_with_string(successors_df)
    return main_concept_df, predecessors_df, successors_df