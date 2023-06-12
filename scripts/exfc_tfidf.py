import scipy.sparse as sp
from drqa import retriever
import json
import pandas as pd
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tfidf_path', type=str, default=None,
                        help='Path to TF-IDF N-grams')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')

    args = parser.parse_args()
    tfrk = retriever.TfidfDocRanker(tfidf_path=args.tfidf_path)

    tfidf_path = 'results/wiki_db-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    tfrk = retriever.TfidfDocRanker(tfidf_path=tfidf_path)
    file_url1 = 'data/data_base.jsonl'
    datares = []
    with open(file_url1, 'r', encoding='utf-8') as fr:
        
        for index,line in enumerate(tqdm(fr)):
            data = json.loads(line)
            tfre, tfsco = tfrk.closest_docs(data['claim'],200)
            data_dict = dict()
            data_dict['claim'] = data['claim']
            data_dict['golden entity'] = data['golden entity']
            data_dict['tfidf rank'] = tfre
            data_dict['tfidf score'] = tfsco
            datares.append(data_dict)
    print('add tfidf rank and score to train/dev/test and save it to additonal csv file')
    data_df = pd.DataFrame(datares)
    data_df = data_df.drop_duplicates(subset='claim')
    data_df = data_df.drop('golden entity', axis=1)
    test = pd.read_csv('/data/mahuanhuan/projects/EX-FEVER/data/test.csv')
    mergedtest_df = pd.merge(test, data_df.drop_duplicates(subset='claim'), on='claim', how='left')
    dev = pd.read_csv('/data/mahuanhuan/projects/EX-FEVER/data/dev.csv')
    mergeddev_df = pd.merge(dev, data_df.drop_duplicates(subset='claim'), on='claim', how='left')
    train = pd.read_csv('/data/mahuanhuan/projects/EX-FEVER/data/train.csv')
    mergedtrain_df = pd.merge(train, data_df.drop_duplicates(subset='claim'), on='claim', how='left')
    # add tfidf rank and score to train/dev/test and save it to additonal csv file
    mergedtest_df.to_csv(f'{args.out_dir}/test_tfrank.csv',index=None)
    mergeddev_df.to_csv(f'{args.out_dir}/dev_tfrank.csv',index=None)
    mergedtrain_df.to_csv(f'{args.out_dir}/train_tfrank.csv',index=None)
