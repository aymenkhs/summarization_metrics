import pandas as pd

import evaluations

def read_eval_data(path):
    data = pd.read_json(path, orient='index')
    # TODO: standerdize the columns with 'prediction' and 'referance'
    data = data[['result_summary', 'true_summary']]
    return data

def main():
    eval_data = read_eval_data('evaluation_data/meaning_cloud_samcorpus_test_results.json')
    eval_data.columns = ['prediction', 'referance']
    evaluations.compute_metrics(eval_data, metrics='all')
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
