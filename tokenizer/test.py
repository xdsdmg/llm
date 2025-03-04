import unittest

import tokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        t = tokenizer.Tokenizer()
        sentences = [
            "在 Python 中，编写单元测试通常使用 unittest 模块，这是 Python 标准库的一部分。以下是一个简单的步骤指南，帮助你编写单元测试。",
        ]
        vocab_len = 372
        t.train(sentences, vocab_len, show_detail=True)

        print("Frequency statistics of each word in the sentences:")
        for k, v in t.stats().items():
            print(f"word: {k.decode("utf-8")}, word in UTF-8: {k}, frequency: {v}")

        tokens = t.tokenize("在 Python 中，编写单元测试通常使用 unittest 模块，这是 Python 标准库的一部分。以下是一个简单的步骤指南，帮助你编写单元测试。")

        res = []
        token_total = len(tokens)
        decode_failed = 0
        for token in tokens:
            try:
                res.append(token.decode('utf-8'))
            except Exception:
                decode_failed += 1
                res.append(token)

        print(f"the total of token: {token_total}, decode failed: {decode_failed}")
        print(res)


if __name__ == "__main__":
    unittest.main()
