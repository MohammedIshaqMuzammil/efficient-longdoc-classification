import json
import os
import glob
import tqdm
import jsonlines
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything

#Crap Libraries
""" from transformers import *
import torch
from transformers.data.processors.utils import InputExample
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
 """

def read_hyperpartisan_data(hyper_file_path):
    """
    Read a jsonl file for Hyperpartisan News Detection data and return lists of documents and labels
    :param hyper_file_path: path to a jsonl file
    :return: lists of documents and labels
    """
    documents = []
    labels = []
    with jsonlines.open(hyper_file_path) as reader:
        for doc in tqdm.tqdm(reader):
            documents.append(doc['text'])
            labels.append(doc['label'])

    return documents, labels

def prepare_hyperpartisan_data(hyper_path='./data/hyperpartisan'):
    """
    Load the Hyperpartisan News Detection data and prepare the datasets
    :param hyper_path: path to the dataset files, {train, dev, test}.jsonl
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(hyper_path):
        raise Exception("Data path not found: {}".format(hyper_path))

    text_set = {}
    label_set = {}
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(hyper_path, split + '.jsonl')
        text_set[split], label_set[split] = read_hyperpartisan_data(file_path)

    enc = preprocessing.LabelBinarizer()
    enc.fit(label_set['train'])
    num_labels = 1 # binary classification
    # vectorize labels as zeros and ones
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.transform(label_set[split])
    return text_set, vectorized_labels, num_labels

def clean_20news_data(text_str):
    """
    Clean up 20NewsGroups text data, from CogLTX: https://github.com/Sleepychord/CogLTX/blob/main/20news/process_20news.py
    // SPDX-License-Identifier: MIT
    :param text_str: text string to clean up
    :return: clean text string
    """
    tmp_doc = []
    for words in text_str.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc

def prepare_20news_data():
    """
    Load the 20NewsGroups datasets and split the original train set into train/dev sets
    :return: dicts of lists of documents and labels and number of labels
    """
    text_set = {}
    label_set = {}
    test_set = fetch_20newsgroups(subset='test', random_state=21)
    text_set['test'] = [clean_20news_data(text) for text in test_set.data]
    label_set['test'] = test_set.target

    train_set = fetch_20newsgroups(subset='train', random_state=21)
    train_text = [clean_20news_data(text) for text in train_set.data]
    train_label = train_set.target

    # take 10% of the train set as the dev set
    text_set['train'], text_set['dev'], label_set['train'], label_set['dev'] = train_test_split(train_text,
                                                                                                train_label,
                                                                                                test_size=0.10,
                                                                                                random_state=21)
    enc = preprocessing.LabelEncoder()
    enc.fit(label_set['train'])
    num_labels = len(enc.classes_)

    # vectorize labels as zeros and ones
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.transform(label_set[split])

    return text_set, vectorized_labels, num_labels

def prepare_eurlex_data(inverted, eur_path='./data/EURLEX57K'):
    """
    Load EURLEX-57K dataset and prepare the datasets
    :param inverted: whether to invert the section order or not
    :param eur_path: path to the EURLEX files
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(eur_path):
        raise Exception("Data path not found: {}".format(eur_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}

    for split in ['train', 'dev', 'test']:
        file_paths = glob.glob(os.path.join(eur_path, split, '*.json'))
        for file_path in tqdm.tqdm(sorted(file_paths)):
            text, tags = read_eurlex_file(file_path, inverted)
            text_set[split].append(text)
            label_set[split].append(tags)

    vectorized_labels, num_labels = vectorize_labels(label_set)

    return text_set, vectorized_labels, num_labels

def read_eurlex_file(eur_file_path, inverted):
    """
    Read each json file and return lists of documents and labels
    :param eur_file_path: path to a json file
    :param inverted: whether to invert the section order or not
    :return: list of documents and labels
    """
    tags = []
    with open(eur_file_path) as file:
        data = json.load(file)
    sections = []
    text = ''
    if inverted:
        sections.extend(data['main_body'])
        sections.append(data['recitals'])
        sections.append(data['header'])

    else:
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])

    text = '\n'.join(sections)

    for concept in data['concepts']:
        tags.append(concept)

    return text, tags

def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None # when genre information is missing

def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                          converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary']) # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test

def prepare_book_summaries(pairs, book_path='data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets
    :param pairs: whether to combine pairs of documents or not
    :param book_path: path to the booksummaries.txt file
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(book_path):
        raise Exception("Data not found: {}".format(book_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    if not pairs:
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]

        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i+1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i+1])))

        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))

        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i+1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i+1])))

    vectorized_labels, num_labels = vectorize_labels(label_set)
    return text_set, vectorized_labels, num_labels

def vectorize_labels(all_labels):
    """
    Combine labels across all data and reformat the labels e.g. [[1, 2], ..., [123, 343, 4] ] --> [[0, 1, 1, ... 0], ...]
    Only used for multi-label classification
    :param all_labels: dict with labels with keys 'train', 'dev', 'test'
    :return: dict of vectorized labels per split and total number of labels
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            all_set.extend(labels)
    all_set = list(set(all_set))

    mlb = MultiLabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)

    print(f'Total number of labels: {num_labels}')

    result = {}
    for split in all_labels:
        result[split] = mlb.transform(all_labels[split])

    return result, num_labels



#PROPER YELP


def vectorize_labels_yelp_old(all_labels):
    """
    Combine labels across all data and reformat the labels e.g. [1, 2, ..., 123, 343, 4] --> [[1, 0, 0], [0, 1, 0], ..., [0, 0, 1]]
    Only used for multi-class classification
    :param all_labels: dict with labels with keys 'train', 'dev', 'test'
    :return: dict of vectorized labels per split and total number of labels
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            if isinstance(labels, float):
                all_set.append(int(labels))
            else:
                all_set.extend([int(label) for label in labels])
    all_set = list(set(all_set))

    mlb = MultiLabelBinarizer()
    mlb.fit_transform([[label] for label in all_set])
    num_labels = len(mlb.classes_)

    print(f'Total number of labels: {num_labels}')

    result = {}
    for split in all_labels:
        if isinstance(all_labels[split], float):
            result[split] = mlb.transform([[int(all_labels[split])]])
        else:
            result[split] = mlb.transform(all_labels[split])

    return result, num_labels



def vectorize_labels_yelp(labels):
    """
    Convert list of labels to binary vectors.
    Only used for multi-class classification
    :param labels: list of labels, where each label is an integer
    :return: binary vectors for the labels
    """
    all_set = []
    for split in labels:
        if split in labels:
            for label in labels[split]:
                if isinstance(label, float):
                    all_set.append(int(label))
                else:
                    all_set.extend([int(l) for l in label])
    all_set = list(set(all_set))

    mlb = LabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)

    print(f'Total number of labels: {num_labels}')

    result = {}
    for split in labels:
        if split in labels:
            result[split] = mlb.transform(labels[split])

    return result, num_labels



def prepare_yelp_data(yelp_path='data/yelp_academic_dataset_review.json', num_samples=20000):
    """
    Load Yelp dataset and prepare the datasets
    :param yelp_path: path to the Yelp JSON file
    :param num_samples: number of samples to load (set to None to load all)
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(yelp_path):
        raise Exception("Data path not found: {}".format(yelp_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}

    with open(yelp_path) as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            if num_samples and i >= num_samples:
                break
            data = json.loads(line)
            text = data['text']
            label = int(data['stars'])
            #split = get_yelp_split(data['split'])
            split = get_split()

            text_set[split].append(text)
            label_set[split].append(label)

    vectorized_labels, num_labels = vectorize_labels_yelp(label_set)

    return text_set, vectorized_labels, num_labels


def get_yelp_split(s):
    if s == 'train':
        return 'train'
    elif s == 'dev':
        return 'dev'
    elif s == 'test':
        return 'test'
    else:
        raise ValueError(f"Invalid split value: {s}") 



def get_split(train_pct=0.8, dev_pct=0.1):
    """
    Get the split for the current example
    :param train_pct: percentage of data to use for training
    :param dev_pct: percentage of data to use for development
    :return: 'train', 'dev', or 'test'
    """
    r = np.random.random()
    if r < train_pct:
        return 'train'
    elif r < train_pct + dev_pct:
        return 'dev'
    else:
        return 'test'




#YELP CRAP
def load_yelpdataset_data(json_reader, book_path):
    """
    Load the Yelp Dataset Summary and split it into train/dev/test sets.
    :param book_path: path to the booksummaries.txt file
    :return train, dev, test as pandas data frames
    """
    nrows=10000
    min_length=100
    max_length=128
    df = None
    while True:
        df_candidate = next(json_reader)
        df_candidate = df_candidate.loc[(df_candidate['text'].str.len() > min_length) & (df_candidate['text'].str.len() <= max_length), ['text', 'stars']]
        if df is None:
            df = df_candidate
        else:
            df = df.append(df_candidate)
        for rating in range(1, 6, 1):
            df_rating = df[df['stars'] == rating]
            if len(df_rating) > nrows//5:
                df_rating = df_rating.iloc[:nrows//5, :]
                df = df.loc[~(df['stars'] == rating), :]
                df = df.append(df_rating)
        if len(df) == nrows:
            return df
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prepare_yelpdataset_data(book_path='./BERT2/yelp_academic_dataset_review.json'):
    reader = pd.read_json(book_path, lines=True, chunksize=10000)
    train_df = load_yelpdataset_data(reader)
    test_df = load_yelpdataset_data(reader)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=200)
    model.classifier.add_module('bert_activation', nn.Tanh())
    model.classifier.add_module('prediction', nn.Linear(200, 5))

    FINE_TUNE = True
    #print(f'Total model trainable parameters {count_parameters(model)}')
    if FINE_TUNE:
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
        #print(f'Total head trainable parameters {count_parameters(model)}')
    model.cuda();

    model.classifier

def prepare_yelpdataset_data2(df, text_col, label_col, tokenizer):
    l = [InputExample(guid=idx, text_a=df.loc[idx, text_col], label=df.loc[idx, label_col]) for 
       idx, row in tqdm(df.iterrows(), total=df.shape[0])]
    features = glue_convert_examples_to_features(examples=l, 
                                    tokenizer=tokenizer,
                                    max_length=300,
                                    label_list = df[label_col].values,
                                    output_mode='regression')

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label-1 for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    return dataset

def prepare_yelpdataset_data3(train_df, test_df):
    train_dataset = prepare_yelpdataset_data2(train_df, 'text', 'stars')
    test_dataset = prepare_yelpdataset_data2(test_df, 'text', 'stars')
    val_idx, train_idx = train_test_split(np.arange(len(train_dataset)), random_state=4, train_size=0.1)
    total_size = len(train_dataset)
    val_dataset = TensorDataset(*train_dataset[val_idx])
    train_dataset = TensorDataset(*train_dataset[train_idx])
    assert total_size == len(val_dataset) + len(train_dataset)





if __name__ == "__main__":
    seed_everything(3456)
    hyper_text_set, hyper_label_set, hyper_num_labels = prepare_hyperpartisan_data()
    assert hyper_num_labels == 1
    assert len(hyper_text_set['train']) == len(hyper_label_set['train']) == 516
    assert len(hyper_text_set['dev']) == len(hyper_label_set['dev']) == 64
    assert len(hyper_text_set['test']) == len(hyper_label_set['test']) == 65
    news_text_set, news_label_set, news_num_labels = prepare_20news_data()
    assert news_num_labels == 20
    assert len(news_text_set['train']) == len(news_label_set['train']) == 10182
    assert len(news_text_set['dev']) == len(news_label_set['dev']) == 1132
    assert len(news_text_set['test']) == len(news_label_set['test']) == 7532
    eur_text_set, eur_label_set, eur_num_labels = prepare_eurlex_data(False)
    assert eur_num_labels == 4271
    assert len(eur_text_set['train']) == len(eur_label_set['train']) == 45000
    assert len(eur_text_set['dev']) == len(eur_label_set['dev']) == 6000
    assert len(eur_text_set['test']) == len(eur_label_set['test']) == 6000
    inverted_text_set, inverted_label_set, inverted_num_labels = prepare_eurlex_data(True)
    assert inverted_num_labels == 4271
    assert len(inverted_text_set['train']) == len(inverted_label_set['train']) == 45000
    assert len(inverted_text_set['dev']) == len(inverted_label_set['dev']) == 6000
    assert len(inverted_text_set['test']) == len(inverted_label_set['test']) == 6000
    book_text_set, book_label_set, book_num_labels = prepare_book_summaries(False)
    assert book_num_labels == 227
    assert len(book_text_set['train']) == len(book_label_set['train']) == 10230
    assert len(book_text_set['dev']) == len(book_label_set['dev']) == 1279
    assert len(book_text_set['test']) == len(book_label_set['test']) == 1279
    pair_text_set, pair_label_set, pair_num_labels = prepare_book_summaries(True)
    assert pair_num_labels == 227
    assert len(pair_text_set['train']) == len(pair_label_set['train']) == 5115
    assert len(pair_text_set['dev']) == len(pair_label_set['dev']) == 639
    assert len(pair_text_set['test']) == len(pair_label_set['test']) == 639

    yelp_text_set, yelp_label_set, yelp_num_labels = prepare_yelp_data('data/yelp_academic_dataset_review.json')
    assert yelp_num_labels == 5
    assert len(yelp_text_set['train']) == len(yelp_label_set['train']) == 16000
    assert len(yelp_text_set['dev']) == len(yelp_label_set['dev']) == 2000
    assert len(yelp_text_set['test']) == len(yelp_label_set['test']) == 2000
    print("MAIN DATALOADER MEOW") #TESTING SUCCESS OF THE ASSERTIONS

