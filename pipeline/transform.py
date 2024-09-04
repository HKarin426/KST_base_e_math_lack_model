# transform.py

import pandas as pd

def merge_data(qa_df, irt_df, taker_irt_df):
    merged_df = pd.merge(qa_df, irt_df, on=['testID', 'assessmentItemID'], how='left')
    merged_df = pd.merge(merged_df, taker_irt_df, on=['learnerID', 'testID'], how='left')

    columns_to_drop = ['_id_x', 'Timestamp_x', '_id_y', 'Timestamp_y', '_id', 'Timestamp', 'learnerProfile_y']
    merged_df = merged_df.drop(columns=columns_to_drop)
    merged_df = merged_df.rename(columns={'learnerProfile_x': 'learnerProfile'})
    merged_df['knowledgeTag'] = pd.to_numeric(merged_df['knowledgeTag'], errors='coerce', downcast='integer')

    return merged_df

def filter_data(merged_df):
    filtered_df = merged_df[merged_df['learnerProfile'].str.split(';').str[-1] == '1']
    return filtered_df
