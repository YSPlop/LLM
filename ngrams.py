import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

sample_text = 'I am learning NLP(Natural Language Processing)'
tokens = word_tokenize(sample_text)

# Unigram
unigrams = list(ngrams(tokens, 1))
print('Unigrams:', unigrams)
print('\n')

# Bigram
bigrams = list(ngrams(tokens, 2))
print('Bigrams:', bigrams)
print('\n')

# Trigram
trigrams = list(ngrams(tokens, 3))
print('Trigrams:', trigrams)
print('\n')
