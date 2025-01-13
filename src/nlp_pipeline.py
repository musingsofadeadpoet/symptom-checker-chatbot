from sklearn.feature_extraction.text import CountVectorizer
import spacy

nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def vectorize(texts, tokenizer=None):
    vectorizer = CountVectorizer(tokenizer=tokenizer or tokenize)
    return vectorizer.fit_transform(texts), vectorizer
