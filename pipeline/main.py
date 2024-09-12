from data import *
from controller import *
import time
from tqdm import tqdm

def main():
    start_time = time.time()  # 전체 실행 시간 측정 시작

    # 데이터 로딩 진행 상황을 업데이트하기 위해 total_steps 설정
    total_steps = 100
    
    with tqdm(total=total_steps, desc='데이터 로딩 진행 상황', ncols=100) as pbar:
        # MongoDB에서 데이터 가져오기
        mongo_start_time = time.time()
        merged_df = load_qa_irt_theta_data()  # MongoDB 데이터를 로드하여 merged_df 생성
        mongo_end_time = time.time()
        mongo_duration = mongo_end_time - mongo_start_time
        pbar.update(total_steps // 3)  # 33% 완료 (1/3)
        
        # PostgreSQL에서 데이터 가져오기
        postgres_start_time = time.time()
        education_data = load_education_data()  # PostgreSQL에서 education_2022 데이터를 로드
        postgres_end_time = time.time()
        postgres_duration = postgres_end_time - postgres_start_time
        pbar.update(total_steps // 3)  # 66% 완료 (2/3)
        
        # label_math 데이터 가져오기
        label_math_start_time = time.time()
        label_math_ele = load_label_math_ele()  # PostgreSQL에서 label_math_ele 데이터를 로드
        label_math_end_time = time.time()
        label_math_duration = label_math_end_time - label_math_start_time
        pbar.update(total_steps - (total_steps // 3 * 2))  # 100% 완료 (3/3)
    
    end_time = time.time()
    total_time = end_time - start_time

    # 데이터 로딩 시간 출력
    print(f"merged_df 데이터 로딩 시간: {mongo_duration:.2f}초")
    print(f"education_data 데이터 로딩 시간: {postgres_duration:.2f}초")
    print(f"label_math_ele 데이터 로딩 시간: {label_math_duration:.2f}초")
    print(f"데이터 로딩 완료. 총 소요 시간: {total_time:.2f}초")
    
    # 데이터 처리 및 병합
    merged_df_with_education = process_merged_data()  # merged_df와 education_data 데이터를 병합한 DataFrame
    
    # 모델 로드
    model = load_model('enhanced_binary_nn_model.pth')  # 모델 로드 함수 호출

    # 학습자 ID 입력
    # 학습자 ID 입력
    learner_id = None
    while True:
        learner_id = input("학습자 ID를 입력하세요: ").strip()
        if learner_id in merged_df_with_education['learnerID'].values:  # 병합된 데이터를 기준으로 학습자 ID 확인
            break
        else:
            print("유효하지 않은 학습자 ID입니다. 다시 입력하세요.")

    # 추천 데이터 생성
    unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping = process_label_math_data(label_math_ele, education_data)

    # 추천 데이터 생성
    prerequisite_recommendations_df, successor_recommendations_df = recommend_all(
        learner_id, merged_df_with_education, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors
    )

    # 추천할 내용이 있는지 확인
    if prerequisite_recommendations_df.empty and successor_recommendations_df.empty:
        print("추천할 이전 학습 개념이 없습니다.")
    else:
        # 추천할 항목이 있으면 출력
        top_prerequisite_recommendations_df = recommend_one_per_area(prerequisite_recommendations_df)
        top_successor_recommendations_df = recommend_one_per_area_successors(successor_recommendations_df, merged_df_with_education)

        print(f"Top recommended prerequisites for learner {learner_id}:")
        print(top_prerequisite_recommendations_df)

        print(f"Top recommended successors for learner {learner_id}:")
        print(top_successor_recommendations_df)

        # 데이터베이스 적재 여부 선택
        while True:
            choice = input("추천 데이터를 데이터베이스에 적재하려면 1번, 적재하지 않으려면 2번을 입력하세요: ").strip()
            if choice == '1':
                # 적재 함수 호출
                recommend_concepts(learner_id, top_prerequisite_recommendations_df, top_successor_recommendations_df)
                print("추천 데이터가 데이터베이스에 적재되었습니다.")
                break
            elif choice == '2':
                print("추천 데이터가 적재되지 않았습니다.")
                break
            else:
                print("잘못된 입력입니다. 1번 또는 2번을 입력하세요.")

if __name__ == "__main__":
    main()
