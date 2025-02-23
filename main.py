from tokenizers import Tokenizer
from tokenizers.models import BPE

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE())

