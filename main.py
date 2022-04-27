import pandas as pd

from metrics import rouge_n, rouge_l

# TODO: remove default evaluation data
def read_eval_data(path = 'evaluation_data/meaning_cloud_samcorpus_test_results.json'):
    data = pd.read_json(path, orient='index')
    # TODO: standerdize the columns with 'prediction' and 'referance'
    data = data[['result_summary', 'true_summary']]
    return data

def main():
    eval_data = read_eval_data()

if __name__ == '__main__':
    main()
