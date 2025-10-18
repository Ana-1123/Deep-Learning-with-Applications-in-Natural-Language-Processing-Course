import regex as re
from collections import defaultdict

class NGramLM:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def preprocess(self, text):
        # Lowercase, remove punctuation, tokenize
        text = text.lower()
        text = re.sub(r'[^\p{L}\s]', '', text, flags=re.UNICODE)
        tokens = text.split()
        return tokens

    def train(self, corpus):
        tokens = self.preprocess(corpus)
        self.vocab.update(tokens)
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i+self.n])
            context = tuple(padded_tokens[i:i+self.n-1])
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def ngram_prob(self, ngram):
        context = ngram[:-1]
        # Laplace smoothing
        count_ngram = self.ngram_counts[ngram] + 1
        count_context = self.context_counts[context] + len(self.vocab)
        return count_ngram / count_context

    def sentence_prob(self, sentence):
        tokens = self.preprocess(sentence)
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        prob = 1.0
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i+self.n])
            prob *= self.ngram_prob(ngram)
        return prob

if __name__ == "__main__":
    with open("romanian_corpus.txt", encoding="utf-8") as f:
        corpus = f.read()

    n = 3  
    model = NGramLM(n)
    model.train(corpus)

    test_sentence = "Genetica este o ramură a biologiei."
    print(f"Probabilitatea propoziției: {model.sentence_prob(test_sentence)}")