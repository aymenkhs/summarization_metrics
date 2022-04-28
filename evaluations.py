from nltk.tokenize import word_tokenize
from nltk.translate import meteor, chrf_score

from bert_score import score

from metrics import rouge_n, rouge_l, ReferanceTooSmallException

def compute_rouge_n(evaluation_data, n=1):
    f_score_list = []
    precision_list = []
    recall_list = []
    for instance in evaluation_data.index:
        try:
            recall, precision, f_score = rouge_n(evaluation_data.loc[instance]['referance'],
                evaluation_data.loc[instance]['prediction'], n=n)
        except ReferanceTooSmallException as e:
            recall, precision, f_score = None, None, None
        f_score_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)

    evaluation_data['rouge_{}_prediction'.format(n)] = precision_list
    evaluation_data['rouge_{}_recall'.format(n)] = recall_list
    evaluation_data['rouge_{}_f_score'.format(n)] = f_score_list

    return evaluation_data


def compute_rouge_l(evaluation_data):
    score = []
    for instance in evaluation_data.index:
        longuest_common_substring = rouge_l(evaluation_data.loc[instance]['referance'],
            evaluation_data.loc[instance]['prediction'])
        score.append(longuest_common_substring)

    evaluation_data['rouge_l'] = score

    return evaluation_data

def compute_rouge_we(evaluation_data):
    pass

def compute_bert_score(evaluation_data):

    predictions = evaluation_data['prediction'].to_list()
    referances = evaluation_data['referance'].to_list()

    precisions, recalls, f_measures = score(predictions, referances, lang="en", verbose=True)

    precisions = [float(val) for val in precisions]
    recalls = [float(val) for val in recalls]
    f_measures = [float(val) for val in f_measures]

    evaluation_data['bert_score_prediction'] = precisions
    evaluation_data['bert_score_recall' ] = recalls
    evaluation_data['bert_score_f_score'] = f_measures

    return evaluation_data


def compute_meteor(evaluation_data):
    meteor_scores = []
    for instance in evaluation_data.index:
        score = meteor([word_tokenize(evaluation_data.loc[instance]['referance'])],
            word_tokenize(evaluation_data.loc[instance]['prediction']))
        meteor_scores.append(score)

    evaluation_data['meteor'] = meteor_scores

    return evaluation_data

def compute_chrf(evaluation_data):
    meteor_scores = []
    for instance in evaluation_data.index:
        score = chrf_score.sentence_chrf(evaluation_data.loc[instance]['referance'],
            evaluation_data.loc[instance]['prediction'])
        meteor_scores.append(score)

    evaluation_data['chrf'] = meteor_scores

    return evaluation_data
