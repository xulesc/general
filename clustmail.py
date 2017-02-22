from __future__ import print_function

import sys
reload(sys)  
sys.setdefaultencoding('Cp1252')

import httplib2
import os

from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

import base64
import email

from random import randint
from time import sleep

from threading import Thread
from Queue import Queue

from elasticsearch import Elasticsearch
from datetime import datetime

import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import re

stemmer = PorterStemmer()

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/gmail-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'DS Tag Cloud'

sources = [
    'dataelixir.com',
    'datainnovation.org',
    'datascienceweekly.org',
    'datacommunitydc.org',
    'kdnuggets',
    'stackexchange.com'
]

# https://developers.google.com/api-client-library/python/apis/gmail/v1
# https://gist.github.com/robulouski/7441883
def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-quickstart.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def get_message_body(message):
    msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
    # http://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not
    b = email.message_from_string(msg_str)
    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
        # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = b.get_payload(decode=True)
    body = body.replace('\r\n',' ')
    #body = re.sub(r'^https?:\/\/.*[\r\n]*', '', body, flags=re.MULTILINE)
    #body = re.sub(r'^http?:\/\/.*[\r\n]*', '', body, flags=re.MULTILINE)    
    body = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?<<>>]))', '', body)
    body = re.sub(r'(?i)\b((?:http?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?<<>>]))', '', body)
    body = re.sub(r'(?i)\b((?:mailto?:\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?<<>>]))', '', body)

    return b['date'], b['from'], body

def put_messages_in_queue(messages, service, q):
    for idx, message in enumerate(messages):
        raw_message = service.users().messages().get(userId='me', id=message['id'], format='raw').execute()
        date, sender, body = get_message_body(raw_message)
        print('Got body of: %d' %idx)
        q.put((idx, message['id'], date, sender, body))

def es_stash_mails(service):

    # https://docs.python.org/2/library/queue.html
    num_worker_threads = 8

    fields = ['idx', 'uid', 'date', 'sender', 'body']

    def do_work(item):
        #idx, uid, date, body = item
        es = Elasticsearch('http://localhost:9200/')
        doc = dict(zip(fields, item))
        date = datetime.strptime(' '.join(doc['date'].split(' ')[:-1]),'%a, %d %b %Y %H:%M:%S')        
        doc['date'] = '{:%Y-%m-%d}'.format(date)
        print('Inserting %s' %doc['uid'])
        try:
            es.index(index='gmail', doc_type='ds', id=doc['uid'], body=doc)
        except:
            print('Oopse! Error inserting %s' %doc['uid'])

    def worker():
        while True:
            item = q.get()
            do_work(item)
            q.task_done()

    q = Queue()
    for i in range(num_worker_threads):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    # read all mails of a label
    # https://developers.google.com/gmail/api/v1/reference/users/messages/list#python
    # http://stackoverflow.com/questions/31921183/how-to-fetch-the-all-mail-label-from-the-gmail-api
    results = service.users().messages().list(userId='me', labelIds=['Label_32']).execute()
    if 'messages' in results:
        put_messages_in_queue(results['messages'], service, q)
        
    while 'nextPageToken' in results:
        page_token = results['nextPageToken']
        results = service.users().messages().list(userId='me', labelIds=['Label_32'], pageToken = page_token).execute()
        put_messages_in_queue(results['messages'], service, q)

    q.join()       # block until all tasks are done

def stem_tokens(tokens, stemmer):
    return map(lambda w: stemmer.stem(w), tokens)

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens

def main():
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    # read all labels
    # https://developers.google.com/gmail/api/v1/reference/users/labels/list
    if False:
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])

        if not labels:
            print('No labels found.')
        else:
            print('Labels:')
            for label in labels:
                print('%s-%s' %(label['name'], label['id']))

    if True:
        es_stash_mails(service)

    # https://gist.github.com/drorata/146ce50807d16fd4a6aa
    es = Elasticsearch('http://localhost:9200/')
    # Initialize the scroll
    page = es.search(
        index = 'gmail',
        doc_type = 'ds',
        scroll = '2m',
        search_type = 'scan',
        size = 1000,
        body = {}
    )
    sid = page['_scroll_id']
    scroll_size = page['hits']['total']

    # Start scrolling
    # http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
    # http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    token_dict = {}
    sender_dict = {}
    while True:
        print("Scrolling...")
        page = es.scroll(scroll_id = sid, scroll = '2m')
        # Update the scroll ID
        sid = page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(page['hits']['hits'])
        if scroll_size == 0:
            print('Empty page found. Exiting')
            break
        print("scroll size: %s" %str(scroll_size))
        # Do something with the obtained page
	#print(page['hits']['hits'][0]['_source']['body'])
        translate_table = dict((ord(char), None) for char in string.punctuation)   
        for doc in page['hits']['hits']:
            text = doc['_source']['body']
            lowers = text.lower()
            # print(type(lowers))
            # no_punctuation = lowers.translate(string.punctuation)
            #no_punctuation = lowers.translate(string.punctuation)
            no_punctuation = lowers.translate(translate_table)
            token_dict[doc['_source']['uid']] = no_punctuation
            sender_dict[doc['_source']['uid']] = re.search(r'[\w\.-]+@[\w\.-]+', doc['_source']['sender']).group(0)
    
    #
    #tokens = tokenize(' '.join(token_dict.values()))
    #e_stopwords = stopwords.words('english')
    #no_stopwords = filter(lambda w: not w in e_stopwords, tokens)
    #count = Counter(no_stopwords)
    #print(count.most_common(50))
    
    #
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    doc_vectors = tfidf.fit_transform(token_dict.values())

    #doc_vectors = tfidf.transform(tfs)
    xy = PCA(n_components=2).fit_transform(doc_vectors.toarray())
    c = KMeans(n_clusters=6).fit_predict(xy)
    
    plt.scatter(xy[:,0], xy[:,1], c=c)
    
    print(np.unique(sender_dict.keys()))

    ## http://pandas.pydata.org/pandas-docs/stable/reshaping.html
    df = pd.DataFrame(np.array([token_dict.keys(), sender_dict.values(), list(c)]).T)
    df.columns = ['uid','sender','cluster']
    
    dfm = df.groupby(['cluster','sender']).count().unstack()
    return df, dfm
    
    

if __name__ == '__main__':
    df, dfm = main()
    
    import seaborn as sns
    sns.heatmap(dfm.T)
    