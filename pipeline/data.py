import pandas as pd

# 데이터 로드 및 전처리
def load_data():
    # merged_df 데이터 로드 및 전처리
    merged_df = pd.read_csv(r'D:\다운로드\merged_data.csv')
    # 불필요한 컬럼 삭제
    columns_to_drop = ['learnerProfile_x', 'learnerProfile_y']
    merged_df = merged_df.drop(columns=columns_to_drop)

    # label_math 데이터 로드 및 전처리
    label_math = pd.read_csv(r'D:\다운로드\label.csv')
    # from_g와 to_g 열 생성
    label_math['from_g'] = label_math['from_semester'].apply(lambda x: x.split('-')[0])
    label_math['to_g'] = label_math['to_semester'].apply(lambda x: x.split('-')[0])
    # 필터링 조건 적용: 고등 포함 / 중등만 포함한 경우 제외
    label_math_ele = label_math[
        ~(
            ((label_math['from_g'] == '고등') | (label_math['to_g'] == '고등')) |
            ((label_math['from_g'] == '중등') & (label_math['to_g'] == '중등'))
        )
    ]
    # 치환을 위한 딕셔너리 생성
    replacement_dict = {
        '중등-중1-1학기': '중등-중7-1학기',
        '중등-중2-2학기': '중등-중8-2학기',
        '중등-중1-2학기': '중등-중7-2학기',
        '중등-중2-1학기': '중등-중8-1학기',
        '중등-중3-1학기': '중등-중9-1학기',
        '중등-중3-2학기': '중등-중9-2학기'
    }
    # 'from_g'가 '중등'인 행들의 'from_semester' 값을 딕셔너리 대로 치환
    label_math_ele.loc[label_math_ele['from_g'] == '중등', 'from_semester'] = label_math_ele['from_semester'].replace(replacement_dict)

    # education_2022 데이터 로드 및 전처리
    file_path = r'D:\다운로드\education_2022_계열화최종.xlsx'
    education_2022 = pd.read_excel(file_path)
    # 'Unnamed: 6' 컬럼을 제거
    education_2022 = education_2022.drop(columns=['Unnamed: 6'], errors='ignore')

    return merged_df, label_math_ele, education_2022
