from nltk.util import ngrams, everygrams

# f-function score
f_score_function = lambda precision, recall : 2 * ((precision * recall) / (precision + recall))

def rouge_n(referance, sentence, n=1):
    referance_ngrams = ngrams(referance.split(), n)
    sentence_ngrams = ngrams(sentence.split(), n)

    referance_ngrams = [item for item in referance_ngrams]
    sentence_ngrams = [item for item in sentence_ngrams]

    matching_ngrams = [item for item in sentence_ngrams if item in referance_ngrams]

    precision = len(matching_ngrams) / len(referance_ngrams)

    try:
        recall = len(matching_ngrams) / len(sentence_ngrams)
        f_score = f_score_function(precision, recall)
    except ZeroDivisionError as e:
        return 0, 0, 0

    return recall, precision, f_score

def rouge_l(reference, sentence):
    referance_ngrams = everygrams(reference.split())
    sentence_ngrams = everygrams(sentence.split())

    referance_ngrams = [item for item in referance_ngrams]
    sentence_ngrams = [item for item in sentence_ngrams]

    matching_ngrams = [item for item in sentence_ngrams if item in referance_ngrams]

    if len(matching_ngrams) > 0:
        longuest_common_substring = len(matching_ngrams[-1])
    else:
        longuest_common_substring = 0

    return longuest_common_substring
