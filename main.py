import sys
import click
from pipeline import controller as C
from pipeline.data import load_data  # 데이터 로드 함수 불러오기

@click.command()
@click.option('-l', '--learner-id', type=click.STRING, required=True, help='분석할 학습자 ID를 입력하세요')
def start_analysis(learner_id):
    # 데이터 로드
    merged_df, _, _ = load_data()  # data.py에서 필요한 데이터 로드

    # 학습자 성과 분석
    correct_knowledge_tags, incorrect_knowledge_tags = C.analyze_student_performance(learner_id, merged_df)
    
    if correct_knowledge_tags is None or incorrect_knowledge_tags is None:
        print(f"학습자 {learner_id}의 데이터가 없어 분석을 종료합니다.")
        sys.exit(1)

    # 가장 빈도수 높은 태그 추출
    most_common_correct_tag = C.get_most_common_tag([correct_knowledge_tags])
    most_common_incorrect_tag = C.get_most_common_tag([incorrect_knowledge_tags])

    # 맞은 문제 관련 본개념, 후속개념 정보 출력
    print(f"\n학습자 {learner_id}의 맞은 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_correct_tag}")
    print("맞은 문제 관련 본개념, 후속개념 정보:")
    correct_main_concept_df, _, correct_successors_df = C.get_concepts_df(most_common_correct_tag)
    print(correct_main_concept_df)
    print(correct_successors_df)
    
    # 틀린 문제 관련 본개념, 선수개념 정보 출력
    print(f"\n학습자 {learner_id}의 틀린 문제에서 가장 빈도수가 높은 knowledgeTag: {most_common_incorrect_tag}")
    print("틀린 문제 관련 본개념, 선수개념 정보:")
    incorrect_main_concept_df, incorrect_predecessors_df, _ = C.get_concepts_df(most_common_incorrect_tag)
    print(incorrect_main_concept_df)
    print(incorrect_predecessors_df)

if __name__ == "__main__":
    start_analysis()
