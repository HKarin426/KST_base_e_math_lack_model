from controller import *

# 데이터 가져오기
merged_df, education_2022, label_math_ele = get_merged_data()


# 본개념, ID-이름 쌍, 학기 정보 생성
unique_id_name_pairs, unique_id_semesters = get_unique_id_name_pairs(label_math_ele)

# 교육 정보 매핑
education_mapping = education_2022.set_index('ID').to_dict('index')

# 선수/후속 개념 관계 생성
predecessors = defaultdict(list)
successors = defaultdict(list)
for _, row in label_math_ele.iterrows():
    from_id = row['from_id']
    to_id = row['to_id']
    predecessors[from_id].append(to_id)
    successors[to_id].append(from_id)

# 사용자 입력 및 분석
learner_id_input = input("학습자 ID를 입력하세요: ").strip()
correct_knowledge_tags, incorrect_knowledge_tags = analyze_student_performance(learner_id_input, merged_df)

if correct_knowledge_tags is not None and incorrect_knowledge_tags is not None:
    most_common_correct_tag = get_most_common_tag([correct_knowledge_tags])
    most_common_incorrect_tag = get_most_common_tag([incorrect_knowledge_tags])
    print(f"\n학습자 {learner_id_input}의 맞은 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_correct_tag}")
    
    correct_main_concept_df, _, correct_successors_df = get_concepts_df(most_common_correct_tag, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters)
    
    print("맞은 문제 관련 본개념 정보:") 
    print(correct_main_concept_df) 
    print("맞은 문제 관련 후속개념 정보:") 
    print(correct_successors_df) 
    
    print(f"\n학습자 {learner_id_input}의 틀린 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_incorrect_tag}")
    
    incorrect_main_concept_df, incorrect_predecessors_df, _ = get_concepts_df(most_common_incorrect_tag, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters)

    print("틀린 문제 관련 본개념 정보:") 
    print(incorrect_main_concept_df) 
    print("틀린 문제 관련 선수개념 정보:") 
    print(incorrect_predecessors_df)