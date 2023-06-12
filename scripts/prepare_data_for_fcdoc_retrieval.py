import argparse
import os
import json
import string
import sqlite3
import collections
import unicodedata
import logging
import hashlib
import pandas as pd

def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return c


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_split",
        default=None,
        type=str,
        required=True,
        help="[train | dev | test]",
    )
    parser.add_argument(
        "--doc_retrieve_range",
        default=20,
        type=int,
        help="Top k tfidf-retrieved documents to be used in neural document retrieval."
    )
    parser.add_argument(
        "--data_dir",
        default='./results',
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default='fc',
        type=str
    )
    parser.add_argument(
        "--oracle",
        action="store_true"
    )

    args = parser.parse_args()
    wiki_db = connect_to_db("data/wiki_db.db")

    args.data_dir = os.path.join(args.data_dir)

    # fc_data = json.load(open(os.path.join('data/', args.data_split+'.json')))
    fc_data = pd.read_csv(os.path.join(args.data_dir, args.data_split+'_tfrank.csv'))
    fc_data_w_tfidf_docs = []

    for index,e in fc_data.iterrows():
        golden_entity = e['golden entity']
        retrieved_docs = e['tfidf rank']
        golden_docs = []
        if type(golden_entity)!=list:
            golden_entity = eval(golden_entity)
        if type(retrieved_docs)!=list:
            retrieved_docs = eval(retrieved_docs)
        for sp in golden_entity:
            if sp not in golden_docs:
                golden_docs.append(sp)
        golden_docs = list(map(lambda x :x.replace('_',' '),golden_docs))
        e['golden entity'] = golden_docs
        e['tfidf rank'] = retrieved_docs
        context, labels = [], []
        for doc_title in retrieved_docs[:args.doc_retrieve_range]:
            if args.oracle and doc_title in golden_docs:
                continue
            para = wiki_db.execute("SELECT * FROM documents WHERE id=(?)", \
                                            ( doc_title,)).fetchall()[0]

            context.append(list(para))
        
            if doc_title in golden_docs:
                labels.append(1)
            else:
                labels.append(0)
        e['context'] = context[:args.doc_retrieve_range]
        e['labels'] = labels[:args.doc_retrieve_range]
        claim = e['claim']
        hass = hashlib.md5(claim.encode(encoding='UTF-8')).hexdigest()
        e['qas_id'] = hass
        e['uid'] = hass
        fc_data_w_tfidf_docs.append(dict(e))

    logging.info("Saving prepared data ...")
    print(f"saving file to {os.path.join(args.data_dir, 'fc_'+args.data_split+'_doc_retrieval.json')}")

    with open(os.path.join(args.data_dir, 'fc_'+args.data_split+'_doc_retrieval.json'), 'w', encoding="utf-8") as f:
        json.dump(fc_data_w_tfidf_docs, f,indent=8)


if __name__ == "__main__":
    main()