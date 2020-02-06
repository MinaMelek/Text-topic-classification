
import sys
import re
import xml.sax.saxutils as saxutils
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#nltk.download('punkt')
#nltk.download('stopwords')


this = sys.modules[__name__]


this.tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
this.lemmatizer = WordNetLemmatizer()
this.vocabulary = ['']
this.categories = []
this.stop_words = set(stopwords.words('english'))

def prepare_data():
    """Added function to read data"""
    df = pd.read_json('../../Database/News_Category_Dataset_v2.json', lines=True)
    #data = df.iloc[:,1:4].drop(columns='date')
    df = df.sample(frac=1,random_state=4).reset_index(drop=True)
    
    #print(cates.size())
    df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    
    #cates = df.groupby('category')
    #print("total categories:", cates.ngroups)
    
    #df_temp = df.copy().drop(columns=['category','headline'])
    df = df.iloc[:,1:4].drop(columns=['date'])
    
    #We only need the Headlines_text column from the data
    data_text = df['headline'].copy()
    
    # We need to remove stopwords first. Casting all values to float will make it easier to iterate over.
    data_text = data_text.astype('str')
    data_text = data_text.apply(lambda s: ' '.join(re.split('\W+', s.lower())).strip()) # remove punctuations
    #for sent in data_text:
    #    add_to_vocab(sent)
    this.vocabulary.append( list(set(data_text.str.cat(sep=' ').split())) )
    X = data_text.to_dict()
    
    target = df.category.values

    labelencoder = LabelEncoder()
    l = list(set(target))
    l.sort()
    this.categories = l ##@
    
    int_category = dict(zip(labelencoder.fit_transform(l), l))
    category_int = dict(zip(l, labelencoder.fit_transform(l)))
    
    target = labelencoder.fit_transform(target)
    Y = np.array(target, dtype='int32')    
    Y = np_utils.to_categorical(Y)
    #Y = dict(enumerate(Y))
    
    seed = 29
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
    x_train = dict(enumerate(x_train))
    y_train = dict(enumerate(y_train))
    x_test = dict(enumerate(x_val))
    y_test = dict(enumerate(y_val))
    
    save_data(x_train, y_train, x_test, y_test, int_category, category_int)

    return (x_train, y_train), (x_test, y_test), int_category, category_int

def generate_categories(reuters=True):
    """Generate the list of categories."""
    topics = './reuters-21578/all-topics-strings.lc.txt'
    if reuters and os.path.exists(topics):
        with open(topics, 'r') as file:
            for category in file.readlines():
                this.categories.append(category.strip().lower())
    else:
        this.categories = np.load('./data/categories.npy')


def vectorize_docs(documents, w2v_model, document_max_num_words=15, num_features=200):
    """A weird oneshot representation for word2vec."""
    this.number_of_documents = len(documents)
    
    x = np.zeros(shape=(this.number_of_documents, document_max_num_words,
                        num_features)).astype(np.float32)

    empty_word = np.zeros(num_features).astype(np.float32)

    for idx, document in enumerate(documents):
        for jdx, word in enumerate(document):
            if jdx == document_max_num_words:
                break

            else:
                if word in w2v_model:
                    x[idx, jdx, :] = w2v_model[word]
                else:
                    x[idx, jdx, :] = empty_word

    return x

def vectorize_idx(documents, w2v_model, document_max_num_words=15):#, num_features=200):
    """A weird oneshot representation for word2vec."""
    number_of_documents = len(documents)
    
    x = np.zeros(shape=(number_of_documents, document_max_num_words)).astype(np.float32)#,num_features)).astype(np.float32)

    # empty_word = 0#np.zeros(num_features).astype(np.float32)

    for idx, document in documents.items():
        for jdx, word in enumerate(document.split()):
            if jdx == document_max_num_words:
                break

            else:
                if word in w2v_model:
                    x[idx, jdx] = w2v_model[word]
                # else:
                #     x[idx, jdx] = empty_word

    return x

def vectorize_categories(categories):
    num_categories = len(this.categories)

    y = np.zeros(shape=(this.number_of_documents, num_categories)).astype(np.float32)

    for idx, key in enumerate(categories.keys()):
        y[idx, :] = categories[key]

    return y


def unescape(text):
    """Unescape charactes."""
    return saxutils.unescape(text)


def unique(arr):
    return list(set(arr))


def add_to_vocab(elements):
    for element in elements.split():
        if element not in this.vocabulary:
            this.vocabulary.append(element)


def add_to_categories(elements):
    for element in elements:
        if element not in this.categories:
            this.categories.append(element)


def transform_to_indices(elements):
    res = []
    for element in elements:
        res.append(this.vocabulary.index(element))
    return res


def transform_to_category_indices(element):
    return this.categories.index(element)


def strip_tags(text):
    """String tags for a better vocabulary."""
    return re.sub('<[^<]+?>', '', text).strip()


def to_category_onehot(categories):
    """Create onehot vectors for categories."""
    target_categories = this.categories
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0

    return vector


def save_data(x_train, y_train, x_test, y_test, int_category, category_int):
#    np.save('./data/x_train_np.npy', x_train)
#    np.save('./data/y_train_np.npy', y_train)
#    np.save('./data/x_test_np.npy', x_test)
#    np.save('./data/y_test_np.npy', y_test)
#    np.savez_compressed('data/All_Data_np.npz',
#                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    filename = os.path.join('data', 'train_test_data.dat')
    with open(filename, 'wb') as handle:
        pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(int_category, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(category_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save('./data/vocabulary.npy', this.vocabulary)
    np.save('./data/categories.npy', this.categories)


def tokenize(document):
    words = []

    for sentence in sent_tokenize(document):
        tokens = [this.lemmatizer.lemmatize(t.lower()) for t in this.tokenizer.tokenize(sentence)
                  if t.lower() not in this.stop_words]
        words += tokens

    return words


def tokenize_docs(document):
    tokenized_docs = []

    for key in document.keys():
        tokenized_docs.append(tokenize(document[key]))

    return tokenized_docs


def read_retuters_files(path="./reuters-21578/"):
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}

    for file in os.listdir(path):
        if file.endswith(".sgm"):
            print("reading ", path + file)
            f = open(path + file, 'r', encoding="ISO-8859-1")
            data = f.read()

            soup = BeautifulSoup(data)
            posts = soup.findAll("reuters")

            for post in posts:
                post_id = post['newid']
                body = unescape(strip_tags(str(post('text')[0].content))
                                .replace('reuter\n&#3;', ''))
                post_categories = []

                topics = post.topics.contents

                for topic in topics:
                    post_categories.append(strip_tags(str(topic)))

                category_onehot = to_category_onehot(post_categories)

                cross_validation_type = post["lewissplit"]
                if (cross_validation_type == "TRAIN"):
                    x_train[post_id] = body
                    y_train[post_id] = category_onehot
                else:
                    x_test[post_id] = body
                    y_test[post_id] = category_onehot

    save_data(x_train, y_train, x_test, y_test)

    return (x_train, y_train), (x_test, y_test)
