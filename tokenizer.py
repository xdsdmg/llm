# Reference: https://zhuanlan.zhihu.com/p/652520262

from typing import Dict, List, Tuple


def pair2bytes(pair: Tuple) -> bytes:
    return pair[0] + pair[1]


class Tokenizer:
    def __init__(self) -> None:
        self.__vocab = [bytes([b]) for b in range(256)]

    def train(self, sentences: List[str], vocab_len: int):
        stats = self.__get_stats(sentences)
        splits = self.__get_splits(stats)

        while len(self.__vocab) < vocab_len:
            pair_freqs = self.__get_pair_freqs(stats, splits)
            best_pair = self.__get_best_pair(pair_freqs)
            if pair_freqs[best_pair] == 1:
                break
            splits = self.__merge_pair(best_pair, splits)
            self.__vocab.append(pair2bytes(best_pair))

    def vocab(self) -> List:
        return self.__vocab

    def __get_stats(self, sentences: List[str]) -> Dict:
        stats = {}

        for sentence in sentences:
            symbols = sentence.split()
            for symbol in symbols:
                k = symbol.encode("utf-8")
                stats[k] = (stats[k] + 1) if k in stats else 1

        return stats

    def __get_splits(self, stats: Dict) -> Dict:
        splits = {word: [bytes([b]) for b in word] for word in stats.keys()}
        return splits

    def __get_pair_freqs(self, stats: Dict, splits: Dict) -> Dict:
        pair_freqs = {}

        for word, freq in stats.items():
            split = splits[word]
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

    def __merge_pair(self, pair: Tuple, splits: Dict):
        for word, split in splits.items():
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if (split[i], split[i + 1]) == pair:
                    split = split[:i] + [pair2bytes(pair)] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split

        return splits


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
