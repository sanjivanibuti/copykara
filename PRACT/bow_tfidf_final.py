from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel

# Input text
text = [
    "I love programming.", 
    "Python is my favorite programming language.", 
    "Programming helps to solve real-life problems."
]

# Tokenize and preprocess the text
tokens = [simple_preprocess(line) for line in text]

# Create the dictionary
dictionary = corpora.Dictionary(tokens)
print("The dictionary has: " + str(len(dictionary)) + " tokens\n")
print("Token-ID mapping:", dictionary.token2id)

# Bag of Words representation
bow_corpus = [dictionary.doc2bow(token) for token in tokens]
print("\nBag of Words Representation:")
for doc in bow_corpus:
    print([[dictionary[id], freq] for id, freq in doc])

# TF-IDF model
tfidf_model = TfidfModel(bow_corpus)

# TF-IDF representation
tfidf_corpus = [tfidf_model[doc] for doc in bow_corpus]
print("\nTF-IDF Representation:")
for doc in tfidf_corpus:
    print(doc)

# Readable TF-IDF representation
print("\nReadable TF-IDF:")
for doc in tfidf_corpus:
    print([[dictionary[id], tfidf_value] for id, tfidf_value in doc])