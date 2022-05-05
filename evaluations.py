import re

from nltk.tokenize import word_tokenize
from nltk.translate import meteor, chrf_score

from bert_score import score

from .rouge_metrics import rouge_n, rouge_l, ReferanceTooSmallException

AVAILABLE_METRICS = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougel',
    'bert_score', 'meteor', 'chrf']

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

    evaluation_data['rouge_{}_precision'.format(n)] = precision_list
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

    evaluation_data['bert_score_precision'] = precisions
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


def compute_metrics(evaluation_data, metrics):

    if metrics.lower() == 'all':
        metrics = AVAILABLE_METRICS
    else:
        metrics = [metric.lower() for metric in metrics if metric.lower() in AVAILABLE_METRICS]

    for metric in metrics:

        print(metric)

        if re.match('rouge[1-9]', metric) is not None:
            evaluation_data = compute_rouge_n(evaluation_data, n=int(metric[5]))
        elif metric == 'rougel':
            evaluation_data = compute_rouge_l(evaluation_data)
        elif metric == 'bert_score':
            evaluation_data = compute_bert_score(evaluation_data)
        elif metric == 'meteor':
            evaluation_data = compute_meteor(evaluation_data)
        elif metric == 'chrf':
            evaluation_data = compute_chrf(evaluation_data)

    return evaluation_data
