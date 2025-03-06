# Reference: https://zhuanlan.zhihu.com/p/652520262
from multiprocessing.reduction import sendfds
from typing import Dict, List, Tuple, Literal
import unicodedata

VOCAB_MIN_SIZE = 256


def pair2bytes(pair: Tuple) -> bytes:
    return pair[0] + pair[1]


class Node:
    def __init__(self, v: int) -> None:
        self.v = v
        self.children = {}


class Trie:
    def __init__(self) -> None:
        self.root = Node(0)

    def insert(self, token: bytes):
        node = self.root

        for b in token:
            if b in node.children:
                node = node.children[b]
            else:
                node.children[b] = Node(b)
                node = node.children[b]

    def get_token(self, sentence: bytes) -> bytes:
        l = 0
        node = self.root
        for b in sentence:
            if b not in node.children:
                break
            l += 1
            node = node.children[b]

        return sentence[:l]


class Tokenizer:
    """ Tokenizer

    A tokenizer implementation based on Byte-level BPE (BBPE).

    Attributes:
        __stats: frequency statistics of each word in the sentences.
        __splits: each byte of the word in `__stats`
    """

    def __init__(self, normalize_func: Literal["NFKC", "NFKD", "NFC", "NFD"] = "NFKC") -> None:
        self.__vocab = [bytes([b]) for b in range(256)]
        self.__trie = Trie()
        self.__stats = {}
        self.__splits = {}
        self.normalize_func = normalize_func

    def vocab(self) -> List[bytes]:
        return self.__vocab

    def splits(self) -> Dict:
        return self.__splits

    def stats(self) -> Dict:
        return self.__stats

    def print_stats(self):
        for k, v in self.__stats.items():
            print(f"word: {k.decode("utf-8")}, word in UTF-8: {k}, frequency: {v}")

    def print_splits(self):
        for k, v in self.__splits.items():
            print(f"word: {k.decode("utf-8")}, word in UTF-8: {k}, tokens: {v}")

    def train(self, sentences: List[str], vocab_len: int, show_detail: bool = False):
        for i in range(len(sentences)):
            sentences[i] = unicodedata.normalize(self.normalize_func, sentences[i])

        if vocab_len < VOCAB_MIN_SIZE:
            raise Exception(
                f"the length of vocab (current value is {vocab_len}) must be larger than {VOCAB_MIN_SIZE}."
            )

        self.__init_stats(sentences)
        self.__init_splits()

        epoch = 0
        while len(self.__vocab) < vocab_len:
            if show_detail:
                print(f"epoch: {epoch}")
                self.print_splits()

            pair_freqs = self.__get_pair_freqs()

            if len(pair_freqs) == 0:
                print(f"Can not find the best pair when the length of the current vocab is {len(self.__vocab)}")
                break

            best_pair = self.__get_best_pair(pair_freqs)

            self.__merge_pair(best_pair)
            self.__vocab.append(pair2bytes(best_pair))

            epoch += 1

        for v in self.__vocab:
            self.__trie.insert(v)

    def tokenize(self, sentence: str) -> List[bytes]:
        tokens = []
        seq = sentence.encode("utf-8")

        while True:
            token = self.__trie.get_token(seq)
            tokens.append(token)
            l = len(token)
            if l >= len(seq):
                break
            seq = seq[l:]

        return tokens

    def __init_stats(self, sentences: List[str]):
        for sentence in sentences:
            symbols = sentence.split()
            for symbol in symbols:
                k = symbol.encode("utf-8")
                self.__stats[k] = (self.__stats[k] + 1) if k in self.__stats else 1

    def __init_splits(self):
        self.__splits = {
            word: [bytes([b]) for b in word] for word in self.__stats.keys()
        }

    def __get_pair_freqs(self) -> Dict:
        pair_freqs = {}

        for word, freq in self.__stats.items():
            split = self.__splits[word]
            l = len(split)

            if l == 1:
                continue

            for i in range(l - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = (
                    pair_freqs[pair] + freq if pair in pair_freqs else freq
                )

        return pair_freqs

    def __get_best_pair(self, pair_freqs: Dict) -> Tuple:
        best_pair = ()
        max_freq = 0

        for pair, freq in pair_freqs.items():
            if freq > max_freq:
                best_pair = pair
                max_freq = freq

        return best_pair

    def __merge_pair(self, pair: Tuple):
        for word, split in self.__splits.items():
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if (split[i], split[i + 1]) == pair:
                    split = split[:i] + [pair2bytes(pair)] + split[i + 2:]
                else:
                    i += 1
            self.__splits[word] = split


if __name__ == "__main__":
    sentences = [
        "我",
        "喜欢",
        "吃",
        "苹果",
        "他",
        "不",
        "喜欢",
        "吃",
        "苹果派",
        "I like to eat apples",
        "She has a cute cat",
        "you are very cute",
        "give you a hug",
    ]

    tokenizer = Tokenizer()
    tokenizer.train(sentences, 300)
    print(tokenizer.vocab())

    print("")

    for k, v in tokenizer.splits().items():
        print(f"{k}, {v}")

    print("")

    for token in tokenizer.tokenize("I like to eat apples"):
        print(token)
