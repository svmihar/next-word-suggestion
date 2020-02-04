from typing import List
from pathlib import Path
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import argparse
from gensim.utils import simple_preprocess
from datetime import datetime
from tqdm import tqdm


def text_to_token(text: List, remove_stopwords=False) -> List:
    return [simple_preprocess(x) for x in text]


    
def train_doc2vec(config, workers=multiprocessing.cpu_count()):
    print(config['text'])
    text_file = open(config['text']).read().splitlines()
    docs = []
    for i, doc in enumerate(text_file):
        docs.append(TaggedDocument(words=simple_preprocess(doc), tags=[i]))
    model = Doc2Vec(vector_size=config['vector_size'], window=config['window'], min_count=config['n_min'], sample=1e-4,
                    negative=5, workers=multiprocessing.cpu_count(), dm=0)
    model.build_vocab(docs)
    for i in tqdm(range(5)):
        model.train(docs, total_examples=model.corpus_count, epochs=20)
        model.min_alpha -= .002
        model.alpha = model.min_alpha
    Path('./models').mkdir(parents=True, exist_ok=True)
    model.save(
        f"./models/{config['output']}_doc2vec_{datetime.now().strftime('%d_%m_%Y')}")
    model.save_word2vec_format(
        f"./models/{config['output']}_word2vec_{datetime.now().strftime('%d_%m_%Y')}")
    return model, docs


if __name__ == "__main__":
    # argparse part
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, type=str, help='txt file')
    ap.add_argument('-o', '--output', required=True,
                    type=str, help='output filename')
    ap.add_argument('-vs', '--vector_size', default=100,
                    required=False, type=int, help='input vector size')
    ap.add_argument('-n', '--n_min', default=1, required=False, type=int, help=('minimal word count'))
    ap.add_argument('-w', '--window', default=10, required=False,
                    type=int, help='input window size')

    config = vars(ap.parse_args())
    train_doc2vec(config)
