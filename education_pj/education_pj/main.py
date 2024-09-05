from controller import *

# 데이터 가져오기
merged_df, education_2022, label_math_ele = get_merged_data()


# # 본개념, ID-이름 쌍, 학기 정보 생성
# unique_id_name_pairs, unique_id_semesters = get_unique_id_name_pairs(label_math_ele)

# # 교육 정보 매핑
# education_mapping = education_2022.set_index('ID').to_dict('index')

# # 선수/후속 개념 관계 생성
# predecessors = defaultdict(list)
# successors = defaultdict(list)
# for _, row in label_math_ele.iterrows():
#     from_id = row['from_id']
#     to_id = row['to_id']
#     predecessors[from_id].append(to_id)
#     successors[to_id].append(from_id)

# # 사용자 입력 및 분석
# learner_id_input = input("학습자 ID를 입력하세요: ").strip()
# correct_knowledge_tags, incorrect_knowledge_tags = analyze_student_performance(learner_id_input, merged_df)

# if correct_knowledge_tags is not None and incorrect_knowledge_tags is not None:
#     most_common_correct_tag = get_most_common_tag([correct_knowledge_tags])
#     most_common_incorrect_tag = get_most_common_tag([incorrect_knowledge_tags])
#     print(f"\n학습자 {learner_id_input}의 맞은 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_correct_tag}")
    
#     correct_main_concept_df, _, correct_successors_df = get_concepts_df(most_common_correct_tag, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters)
    
#     print("맞은 문제 관련 본개념 정보:") 
#     print(correct_main_concept_df) 
#     print("맞은 문제 관련 후속개념 정보:") 
#     print(correct_successors_df) 
    
#     print(f"\n학습자 {learner_id_input}의 틀린 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_incorrect_tag}")
    
#     incorrect_main_concept_df, incorrect_predecessors_df, _ = get_concepts_df(most_common_incorrect_tag, education_mapping, predecessors, successors, unique_id_name_pairs, unique_id_semesters)

#     print("틀린 문제 관련 본개념 정보:") 
#     print(incorrect_main_concept_df) 
#     print("틀린 문제 관련 선수개념 정보:") 
#     print(incorrect_predecessors_df)


import time
from tqdm import tqdm
from collections import defaultdict
from controller import *

def main():
    start_time = time.time()
    
    # 총 단계 수를 100으로 설정하여 1% 단위로 진행 상황을 업데이트
    total_steps = 100
    
    with tqdm(total=total_steps, desc='데이터 로딩 진행 상황', ncols=100) as pbar:
        # MongoDB에서 데이터 가져오기
        mongo_start_time = time.time()
        merged_df = fetch_mongo_data()
        mongo_end_time = time.time()
        mongo_duration = mongo_end_time - mongo_start_time
        pbar.update(total_steps // 3)  # 33% 완료 (1/3)
        
        # PostgreSQL에서 데이터 가져오기
        postgres_start_time = time.time()
        education_2022 = fetch_postgres_data()
        postgres_end_time = time.time()
        postgres_duration = postgres_end_time - postgres_start_time
        pbar.update(total_steps // 3)  # 66% 완료 (2/3)
        
        # label_math 데이터 가져오기
        label_math_start_time = time.time()
        label_math_ele = fetch_label_math()
        label_math_end_time = time.time()
        label_math_duration = label_math_end_time - label_math_start_time
        pbar.update(total_steps - (total_steps // 3 * 2))  # 100% 완료 (3/3)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"merged_data 데이터 로딩 시간: {mongo_duration:.2f}초")
    print(f"education_2022 데이터 로딩 시간: {postgres_duration:.2f}초")
    print(f"label_math_ele 데이터 로딩 시간: {label_math_duration:.2f}초")
    print(f"데이터 로딩 완료. 총 소요 시간: {total_time:.2f}초")
    
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
    while True:
        learner_id_input = input("학습자 ID를 입력하세요: ").strip()
        
        if not learner_id_input:
            print('학습자 ID를 입력하지 않았습니다. 다시 입력해주세요.')
        else:
        # Assuming `merged_df` is available from the `get_merged_data` function
            if learner_id_input not in merged_df['learnerID'].values:
                print('학습자 ID가 존재하지 않습니다.')
            else:
            # Proceed with further processing
                print(f'학습자 ID {learner_id_input} 확인 완료.')
                break
            
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

if __name__ == "__main__":
    main()
