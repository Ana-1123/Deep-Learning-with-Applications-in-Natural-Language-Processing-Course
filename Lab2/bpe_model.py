from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


class SimpleBPE:
    def __init__(self, pretokenizer_name: str = "gpt2", special_tokens: List[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretokenizer_name)
        self.word_freqs: Dict[str, int] = {}
        self.alphabet: List[str] = []
        self.vocab: List[str] = []
        self.merges: Dict[Tuple[str, str], str] = {}
        self.splits: Dict[str, List[str]] = {}
        self.special_tokens = special_tokens or ["<|endoftext|>"]

    # Pre-tokenization & counts 
    def pretokenize_and_count(self, corpus: List[str]):
        """ Pre-tokenize corpus with the pre-tokenizer and compute word frequencies. """
        freqs = defaultdict(int)
        for text in corpus:
            # pre_tokenize_str returns list of (token_str, (start,end))
            tokens_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            words = [w for w, _ in tokens_with_offsets]
            for w in words:
                freqs[w] += 1
        self.word_freqs = freqs

    # initialize char-level vocabulary
    def build_alphabet_and_splits(self):
        """ Build base alphabet from observed characters and initialize character splits for each word. """
        alphabet = []
        for word in self.word_freqs.keys():
            for ch in word:
                if ch not in alphabet:
                    alphabet.append(ch)
        alphabet.sort()
        self.alphabet = alphabet
        # initial vocab: special tokens + alphabet
        self.vocab = list(self.special_tokens) + self.alphabet.copy()
        # initialize splits: each word -> list of characters
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

    # pair frequency computation
    def compute_pair_freqs(self) -> Dict[Tuple[str, str], int]:
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) <= 1:
                continue
            # count adjacent pairs
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    # merge application
    def merge_pair_in_splits(self, a: str, b: str):
        """ Apply a merge (a,b) -> a+b across all splits. """
        merged = a + b
        for word in list(self.splits.keys()):
            split = self.splits[word]
            if len(split) <= 1:
                continue
            i = 0
            new_split = []
            # iterate and do merges greedily left-to-right
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            self.splits[word] = new_split

    # training loop
    def train_bpe(self, corpus: List[str], target_vocab_size: int = 100):
        """
        Train merges until vocabulary reaches target_vocab_size.
        :param corpus: list of strings
        :param target_vocab_size: integer target vocab size (includes special tokens and characters)
        """
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be positive")

        # Step 1: pre-tokenize + counts
        self.pretokenize_and_count(corpus)
        # Step 2: build alphabet and char splits
        self.build_alphabet_and_splits()

        # If already big enough, nothing to do
        while len(self.vocab) < target_vocab_size:
            pair_freqs = self.compute_pair_freqs()
            if not pair_freqs:
                # no pair to merge
                break

            # choose best pair by frequency
            # deterministic tie-breaking: sort pairs by (-freq, pair)
            best_pair, best_freq = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))
            # If best frequency is 0 or None, break
            if best_freq <= 0:
                break

            # perform merge
            a, b = best_pair
            merged = a + b
            self.merge_pair_in_splits(a, b)
            # record merge and update vocab
            self.merges[best_pair] = merged
            # append merged token to vocab if not present
            if merged not in self.vocab:
                self.vocab.append(merged)

    #  tokenization
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single input string using pre-tokenizer and learned merges (in the learned order).
        Returns list of tokens (strings).
        """
        # pre-tokenize using the same pre-tokenizer
        pre = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        words = [w for w, _ in pre]
        # split each word into characters
        splits = [[c for c in w] for w in words]
        # apply merges in the order they were learned
        # merges is dict with keys as tuple pairs; order of iteration should be insertion order (learned order)
        for pair, merged in self.merges.items():
            a, b = pair
            for idx, split in enumerate(splits):
                if len(split) <= 1:
                    continue
                i = 0
                new_split = []
                while i < len(split):
                    if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[idx] = new_split
        # flatten
        tokens = [tok for word in splits for tok in word]
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Example detokenize: just glue tokens back together. This is naive and does not perfectly match GPT-2 detokenization."""
        return "".join(tokens)


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    bpe = SimpleBPE(pretokenizer_name="gpt2", special_tokens=["<|endoftext|>"])
    print("Training BPE... (this uses GPT-2 pre-tokenizer for whitespace handling)")
    bpe.train_bpe(corpus, target_vocab_size=50)

    print("\nLearned merges (in order):")
    for i, (pair, merged) in enumerate(bpe.merges.items()):
        print(f"{i+1:02d}: {pair} -> {merged}")

    print("\nVocabulary snapshot (first 60 tokens):")
    print(bpe.vocab[:60])

    # Tokenize a new sentence
    sample = "This is not a token."
    tokens = bpe.tokenize(sample)
    print(f"\nInput: {sample}")
    print("Tokens:", tokens)

    # Detokenize
    print("Detokenized:", bpe.detokenize(tokens))
