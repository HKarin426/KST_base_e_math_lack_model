
# main.py

from extract import extract_data
from transform import merge_data, filter_data
from load import load_filtered_data
from controller import *

def main():
    # 데이터 추출
    qa_df, irt_df, taker_irt_df = extract_data()

    # 데이터 병합 및 전처리
    merged_df = merge_data(qa_df, irt_df, taker_irt_df)
    filtered_df = filter_data(merged_df)

    # PostgreSQL에 데이터 적재
    load_filtered_data(filtered_df)

    # 개념 분석 수행
    label_math_ele = pd.read_csv('label_math_ele.csv')
    education_2022 = pd.read_csv('education_2022.csv')

    label_math_ele = preprocess_data(label_math_ele)
    unique_id_name_pairs, unique_id_semesters = get_unique_id_name_pairs(label_math_ele)
    education_mapping = create_education_mapping(education_2022)
    predecessors, successors = create_predecessors_successors(label_math_ele)

    learner_id_input = 'sample_learner_id'  # 실제 학습자 ID로 대체 필요
    correct_knowledge_tags, incorrect_knowledge_tags = analyze_student_performance(learner_id_input, filtered_df)

    if correct_knowledge_tags is not None and incorrect_knowledge_tags is not None:
        # 결과에서 가장 빈도수 높은 태그 추출
        most_common_correct_tag = get_most_common_tag([correct_knowledge_tags])
        most_common_incorrect_tag = get_most_common_tag([incorrect_knowledge_tags])

        print(f"\n학습자 {learner_id_input}의 맞은 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_correct_tag}")
        print("맞은 문제 관련 본개념, 후속개념 정보:")
        correct_main_concept_df, _, correct_successors_df = get_concepts_df(most_common_correct_tag)  # 선수학습을 무시하기 위해 '_' 사용
        print(correct_main_concept_df)
        print(correct_successors_df)

        # 틀린 문제의 knowledgeTag로 개념 정보 출력
        print(f"\n학습자 {learner_id_input}의 틀린 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_incorrect_tag}")
        print("틀린 문제 관련 본개념, 선수개념 정보:")
        incorrect_main_concept_df, incorrect_predecessors_df, _ = get_concepts_df(most_common_incorrect_tag)  # 후속학습을 무시하기 위해 '_' 사용
        print(incorrect_main_concept_df)
        print(incorrect_predecessors_df)

if __name__ == "__main__":
    main()



'''

from extract import extract_data
from transform import merge_data, filter_data
from load import load_filtered_data
from controller import *

def main():
    # 데이터 추출
    qa_df, irt_df, taker_irt_df = extract_data()

    # 데이터 병합 및 전처리
    merged_df = merge_data(qa_df, irt_df, taker_irt_df)
    filtered_df = filter_data(merged_df)

    # PostgreSQL에 데이터 적재
    load_filtered_data(filtered_df)

    # 개념 분석 수행
    label_math_ele = pd.read_csv('label_math_ele.csv')
    education_2022 = pd.read_csv('education_2022.csv')

    label_math_ele = preprocess_data(label_math_ele)
    unique_id_name_pairs, unique_id_semesters = get_unique_id_name_pairs(label_math_ele)
    education_mapping = create_education_mapping(education_2022)
    predecessors, successors = create_predecessors_successors(label_math_ele)

    learner_id = 'sample_learner_id'  # 실제 학습자 ID로 대체 필요
    correct_tags, incorrect_tags = analyze_student_performance(learner_id, filtered_df)

    most_common_correct_tag = get_most_common_tag([correct_tags])
    most_common_incorrect_tag = get_most_common_tag([incorrect_tags])

    node_id = 'sample_node_id'  # 실제 노드 ID로 대체 필요
    concepts = get_concepts(node_id, unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping)

    print(f'학습자가 가장 많이 정답을 맞춘 지식 태그: {most_common_correct_tag}')
    print(f'학습자가 가장 많이 오답을 낸 지식 태그: {most_common_incorrect_tag}')
    print(f'개념 정보: {concepts}')

if __name__ == "__main__":
    main()
    '''
