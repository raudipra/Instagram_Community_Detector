## Cluster Community By Num of Cluster ##
from nltk.corpus import stopwords
import numpy as np
import string
from sklearn.pipeline import Pipeline
import json
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics import pairwise_distances
import pickle
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import sys

## Initialize ##
stop = stopwords.words('english') + list(string.punctuation)
symbol = ",_-=+\'\"?/&*:;~`^"
prohibited_tag = ['\'\'',',','(',')','--',':','.','CC','DT','EX','FW','IN','MD','PDT','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','WDT','WP','WP$','WRB']
wnl = WordNetLemmatizer()
ls = LancasterStemmer()
def sentence_to_tokens(sentence):
        tokens = word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        return [ls.stem(x[0]) for x in tags if (x[1] not in prohibited_tag) and (x[0] not in stop) and (x[0][0] not in symbol) and (x[0][len(x[0])-1] not in symbol)]

token_idxs = {}
subreddit_idxs = {}
count_token = 0
count_subreddit = 0

## Build Token and Subreddit, Pake Sample Data Aja yang ada semua komunitas ##
with open(sys.argv[1]) as f:
    count = 0
    for line in f:
        data = json.loads(line)
        sentence = re.sub(r"http\S+", "", data[1])
        sentence = re.sub(r'[^\w]', ' ', sentence)
        sentence = re.sub(r"//\S+", "", sentence)
        subreddit = data[0]
        
        if not subreddit in subreddit_idxs:
            subreddit_idxs[subreddit] = count_subreddit
            count_subreddit += 1
        
        for token in sentence_to_tokens(sentence):
            if not token in token_idxs:
                token_idxs[token] = count_token
                count_token += 1
        count += 1
        if (count % 100000) == 0:
            print(count)

print("Jumlah Token : "+str(count_token))
print("Jumlah Subreddit : "+str(count_subreddit))

## Build Frequent Term Document Matrix ##
frequent_term_document_matrix = {}

with open(sys.argv[1]) as f:
    count = 0
    for line in f:
        count += 1
        data = json.loads(line)
        sentence = re.sub(r"http\S+", "", data[1])
        sentence = re.sub(r'[^\w]', ' ', sentence)
        sentence = re.sub(r"//\S+", "", sentence)
        subreddit = data[0]
        subreddit_idx = subreddit_idxs[subreddit]

        for token in sentence_to_tokens(sentence):
            token_idx = token_idxs[token]
            if frequent_term_document_matrix.get(token_idx) != None:
                if frequent_term_document_matrix[token_idx].get(subreddit_idx) != None:
                    frequent_term_document_matrix[token_idx][subreddit_idx] += 1
                else:
                    frequent_term_document_matrix[token_idx][subreddit_idx] = 1
            else:
                frequent_term_document_matrix[token_idx] = {}
                frequent_term_document_matrix[token_idx][subreddit_idx] = 1
        if (count % 100000) == 0:
            print(count)

# Hapus Variable
token_idxs = None

## Build Sparse Matrix of FTDM ##
row = []
col = []
data = []
for key, value in frequent_term_document_matrix.items():
    for key1, value1 in value.items():
        row.append(float(key))
        col.append(float(key1))
        data.append(float(value1))

# Hapus Variable
frequent_term_document_matrix = None
sparse_matrix = csr_matrix((data, (row, col)), shape=(count_token, count_subreddit))
# Hapus Variable
row = []
col = []
data = []
# print sparse_matrix.toarray()
# print sparse_matrix
# print sparse_matrix.shape, count_subreddit

## Build SVD Form of FTDM Sparse ##
U, s, V = svds(sparse_matrix,k=count_subreddit-1,which='LM')
# print U, s, V
S = np.zeros((count_subreddit-1, count_subreddit-1), dtype=complex)
S[:count_subreddit-1, :count_subreddit-1] = np.diag(s)
s = None
# print np.allclose(sparse_matrix.toarray(), np.dot(U, np.dot(S, V)))
# print sparse_matrix.toarray()
# print np.dot(U, np.dot(S, V))
S = S[:count_subreddit-1, :count_subreddit-1]
U = U[:,:count_subreddit-1]
V = np.transpose(V)
print(U.shape, S.shape, V.shape)

## Cluster Community ##
## CAN, INI NCOMMUNITY BEBAS MO MASUKIN BERAPA ##
ncommunity = int(sys.argv[2])
new_communities = {}
for i in range(count_subreddit):
    new_communities[i] = i
    
## Hitung Similarity Satu Sama Lain ##
distance_matrix = pairwise_distances(V, V, metric='cosine', n_jobs=1)
print(distance_matrix)

## Masukin ke Array Lalu Sort ##
arr = []
for i in range(len(distance_matrix)):
    arr += [x for x in distance_matrix[i]]
arr = sorted(arr)

## Hapus Similarity Antar Dokumen Yang Sama (Similarity Dokumen 1 dan Dokumen 1 Pasti 0) ##
del arr[0:count_subreddit]

## Merge Subreddit/Community ##
for i in range(count_subreddit-ncommunity):
    found = False
    while(not(found)):
        for j in range(count_subreddit):
            for k in range(count_subreddit):
                if distance_matrix[j][k] == arr[0]:
                    found = True
                    del arr[0]
                    distance_matrix[j][k] = -1
                    distance_matrix[k][j] = -1
                    cluster_k = [key for key, value in new_communities.items() if value == new_communities.get(k)]
                    for l in cluster_k:
                        new_communities[l] = new_communities.get(j)
            if found:
                break
        if not(found):
            del arr[0]

## Bikin Nomer Cluster Urut Terkecil ##
print("New Communities : Before")
print(new_communities)
temp = []
change = {}
for key, value in new_communities.items():
    temp.append(value)
temp = sorted(temp)
counter = 0
for i in range(len(temp)):
    if (temp[i]!=counter) and (change.get(temp[i])==None):
        change[temp[i]] = counter
        counter += 1
    elif (temp[i]==counter) and (change.get(temp[i])==None):
        counter += 1
print("Perubahan")
print(change)

for key, value in new_communities.items():
    if change.get(value) != None:
        new_communities[key] = change.get(value)
# Hapus Variable
change = None
temp = None
arr = None
print("New Communities : After")
print(new_communities)
count_subreddit = ncommunity

file = open(sys.argv[3],"w")
file.write(str(subreddit_idxs)+"\n")
file.write("Format -> community : cluster\n")
file.write(str(new_communities)+"\n")
file.close()
