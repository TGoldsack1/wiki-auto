'''
Convert the gold standard summaries and the summaries output from out experiments
into the format required to run the pretrained alignment algorithm on them.

Authors: Tomas + Zhihao
'''


import json, os, ast, re
from nltk.tokenize import sent_tokenize, word_tokenize

# Set this to select model output
input_json = "./simplified_my_longsumm_test_AIC_my_pretrained_allenai_output.json"


input_content = open(input_json, "r")
input_lines = input_content.readlines()
input_dicts = [ast.literal_eval(line) for line in input_lines]

print(len(input_lines))

output_dir_path = "./" + input_json.split("/")[-1].split(".")[0]

if not os.path.isdir(output_dir_path):
  os.mkdir(output_dir_path)

# For each test instance, create a test.src file containing gold standard sentences 
# (would usually be tthe complex sentences) and a test.dst file containing the 
# sentences output from our model
for dictionary in input_dicts:
  ground_truth_sents = sent_tokenize(dictionary['ground_truth'])
  simplified_sents = sent_tokenize(dictionary['simplified_prediction'])

  ground_truth_sents = [' '.join(word_tokenize(sent)) + "\n" for sent in ground_truth_sents]
  simplified_sents = [' '.join(word_tokenize(sent)) + "\n" for sent in simplified_sents]

  output_inst_dir_path = output_dir_path + "/" + str(dictionary['index'])

  if not os.path.isdir(output_inst_dir_path):
    os.mkdir(output_inst_dir_path)

  with open(output_inst_dir_path + "/test.src", "w") as gt_output:
    gt_output.writelines(ground_truth_sents)

  with open(output_inst_dir_path + "/test.dst", "w") as simp_output:
    simp_output.writelines(simplified_sents)
  