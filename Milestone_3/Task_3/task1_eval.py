# we need to just iterate through examples
import sys
# sys.path.append("./baseline/")
import task1_infer_2 as task1_infer
from tqdm import tqdm
import numpy as np

input_file = sys.argv[1]

with open(input_file, 'r') as f:
    input_data = f.read().split("\n")

# pipe delimited knowledge, history and pred_response
input_data = [x.split("|") for x in input_data]
header = input_data[0]
input_data = input_data[1:]

scores = []
for fields in tqdm(input_data):
    if len(fields) == 3:
        knowledge, history, pred_response = fields
    elif len(fields) == 4:
        knowledge, history, pred_response, org_resp = fields
    else:
        knowledge, history, pred_response, org_resp, gt_resp = fields
    # print(len(knowledge), len(history), len(pred_response))
    if len(knowledge) == 0 or len(history) ==0 or len(pred_response) == 0:
        # ignore such cases...
        fields.append("")
        continue 
    res = task1_infer.predict_hallucination(task1_infer.model, knowledge, pred_response, history)
    fields.append(f"{res:.3f}")

# print("*"*20)
# scores = np.array(scores)
# THRESH = 0.7

# acc = (scores > THRESH).mean()
# mean_score = scores.mean()
# print(f'% Hallucinated: {acc}, mean_score: {mean_score}')

# saving back the scores...
new_header = header+["critic_score"]
content = new_header + input_data
with open(input_file, 'w') as f:
    f.write("\n".join(["|".join(x) for x in content]))
