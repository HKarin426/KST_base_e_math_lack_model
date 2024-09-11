from data import *
from controller import *

def main():
    # 데이터 로드
    merged_df, label_math_ele = load_data()
    education_data = load_education_data()  # education_2022 로드
    print("Merged DataFrame head:\n", merged_df.head())
    print("Education DataFrame head:\n", education_data.head())
    print("Label Math DataFrame head:\n", label_math_ele.head())

    # 모델 로드
    model = load_model('enhanced_binary_nn_model.pth')  # 모델 로드 함수 호출

    # 학습자 ID 입력
    # 학습자 ID 입력
    learner_id = None
    while True:
        learner_id = input("학습자 ID를 입력하세요: ").strip()
        if learner_id in merged_df['learnerID'].values:
            break
        else:
            print("유효하지 않은 학습자 ID입니다. 다시 입력하세요.")

    # 추천 데이터 생성
    unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping = process_label_math_data(label_math_ele, education_data)

    # 추천 데이터 생성
    prerequisite_recommendations_df, successor_recommendations_df = recommend_all(
        learner_id, merged_df, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors
    )

    # 추천할 내용이 있는지 확인
    if prerequisite_recommendations_df.empty and successor_recommendations_df.empty:
        print("추천할 이전 학습 개념이 없습니다.")
    else:
        # 추천할 항목이 있으면 출력
        top_prerequisite_recommendations_df = recommend_one_per_area(prerequisite_recommendations_df)
        top_successor_recommendations_df = recommend_one_per_area_successors(successor_recommendations_df, merged_df)

        print(f"Top recommended prerequisites for learner {learner_id}:")
        print(top_prerequisite_recommendations_df)

        print(f"Top recommended successors for learner {learner_id}:")
        print(top_successor_recommendations_df)

if __name__ == "__main__":
    main()