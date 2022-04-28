from metrics import rouge_n, rouge_l

def compute_rouge_n(evaluation_data, n=1):
    f_score_list = []
    precision_list = []
    recall_list = []
    for instance in evaluation_data.index:
        recall, precision, f_score = rouge_n(evaluation_data.loc[instance]['referance'],
            evaluation_data.loc[instance]['prediction'], n=n)
        f_score_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)

    evaluation_data['rouge_{}_prediction'.format(n)] = precision_list
    evaluation_data['rouge_{}_recall'.format(n)] = recall_list
    evaluation_data['rouge_{}_f_score'.format(n)] = f_score_list

    return evaluation_data


def compute_rouge_l(evaluation_data):
    pass

def compute_rouge_we(evaluation_data):
    pass

def compute_bert_score(evaluation_data):
    pass

def compute_meteor(evaluation_data):
    pass

def compute_chrf(evaluation_data):
    pass
