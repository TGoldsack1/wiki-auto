from __future__ import division

from model import *
from util import *
from datetime import timedelta
from datetime import datetime
import time
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

import argparse

def main(args):
    print("start running")

    lsents_datafile = open(args.test_gold, "r")
    test_lsents = lsents_datafile.readlines()

    rsents_datafile = open(args.test_real, "r")
    test_rsents = rsents_datafile.readlines()

    # Bert related
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

    device = torch.device("cuda" )

    tokenizer = tokenizer_class.from_pretrained(args.BERT_folder,
                                                do_lower_case=True)
    model = model_class.from_pretrained(args.BERT_folder, \
                                        output_hidden_states=True)
    model.to(device)
    
    
    for test_i in range(len(test_lsents[:1])):

        output_type, output_score, predect_sequence = model(test_lsents[test_i], test_rsents[test_i], None)

        print(test_lsents[test_i])
        print(test_rsents[test_i])
        print(predect_sequence)

    #     sub_prediected_alignment = []

    #     for i in range(len(predect_sequence)):

    #         if predect_sequence[i] != 0:
    #             small_pair = ["simple_{}".format(i), "complex_{}".format(predect_sequence[i] - 1)]
    #             sub_prediected_alignment.append([paragraph_alignments[test_i]["new_to_original"][small_pair[0]], \
    #                                              paragraph_alignments[test_i]["new_to_original"][small_pair[1]]])

    #     predicted_alignment.extend(sub_prediected_alignment)

    # tmptmp_predicted_alignment = [[i[1], i[0]] for i in predicted_alignment]


    # bert_for_sent_seq_model.eval()
    # Bert related end

    #print('Testing set size: %d' % len(test_set[0]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--test_gold', type=str, default='', required=True, help='Path to the gold test data.')

    parser.add_argument('--test_real', type=str, default='', required=True, help='Path to the real test data.')

    parser.add_argument('--BERT_folder', type=str, default='', required=True, help='Path to the fine-tuned BERT folder.')

    args = parser.parse_args()
    main(args)