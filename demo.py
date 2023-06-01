import json
import pandas as pd
from conclude_detector import init_conclude_model, conclude_detector

# set paths
pattern_path = 'data/conclude_pattern_121.txt'
model_path = 'data/gbdt_conclude_detector.plk'
patterns = open(pattern_path, 'r').read().strip().split('\n')
embedding_path = 'data/qq_w2v.pickle'
pattern2index_path = 'data/conclude_pattern2index.plk'
# init conclude_model
conclude_model = init_conclude_model(
    model_path, patterns, pattern2index_path, embedding_path)
# load text_list
df_course = pd.read_excel('test_data/demo.xlsx')
text_list = json.loads(
    df_course[['begin_time', 'end_time', 'text']].to_json(orient='records'))
# 调用检测
result = conclude_detector(text_list,conclude_model)
print(result)