"""
BPE encoder
This encoder is based on the implementation from OpenAI GPT-2
for the encode/decoder functions and the idea of byte-level BPE tokenization.
The difference is that it already takes split on words text as input.
Fit function for encoder is based on the paper:
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
(Sennrich et al., ACL 2016)
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from logger import logger

Word = Tuple[str, ...]
Pair = Tuple[str, str]


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Word) -> Set[Pair]:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def count_pairs(word: Word, pairs_counts: Dict[Pair, int], freq: int = 1):
    """Updates dictionary of symbol pairs counts with counts in a word, multiplied by freq.
    freq is a frequency of a word in a corpus.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    prev_char = word[0]
    for char in word[1:]:
        pairs_counts[(prev_char, char)] += freq
        prev_char = char


def find_and_replace(
    word: Word, pair: Pair, collect_starts: bool = False
) -> Tuple[Word, List[int]]:
    first, second = pair
    new_word = []
    starts = []
    i = 0
    while i < len(word):
        try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
        except ValueError:
            new_word.extend(word[i:])
            break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
            if collect_starts:
                starts.append(i)
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    new_word = tuple(new_word)

    return new_word, starts


def update_counts(counts: Dict[Pair, int], pair: Pair, freq: int):
    if pair not in counts:
        return
    if counts[pair] <= freq:
        counts.pop(pair)
    else:
        counts[pair] -= freq


def update_pairs_counts_from_merge(
    word: Word,
    freq: int,
    pairs_counts: Dict[Pair, int],
    starts: List[int],
    verbose: bool = False,
):
    for k, i in enumerate(starts):
        if verbose:
            logger.info(f"word: {word}, i: {i}, freq: {freq}")
        # if pair is not at the start:
        # update count of pair that ends with first symbol of the pair
        if i > 0:
            if not (k - 1 >= 0 and starts[k - 1] == i - 2):
                # decrease count of pair the ends with first symbol of the pair
                if verbose:
                    logger.info(f"update:{(word[i - 1], word[i])}")
                update_counts(pairs_counts, (word[i - 1], word[i]), freq)

                # increase count of pair that ends with the whole pair
                if verbose:
                    logger.info(f"add:{(word[i - 1], word[i] + word[i + 1])}")
                pairs_counts[(word[i - 1], word[i] + word[i + 1])] += freq

        # if pair is not end of the word:
        # update count of pair that starts with the last symbol of the pair
        if i + 2 < len(word):
            # decrease count of pair the starts with second(last) symbol of the pair
            if verbose:
                logger.info(f"update:{(word[i + 1], word[i + 2])}")
            update_counts(pairs_counts, (word[i + 1], word[i + 2]), freq)

            if not (k + 1 < len(starts) and starts[k + 1] == i + 2 and i + 3 < len(word)):
                # increase count of pair that starts with the whole pair
                if verbose:
                    logger.info(f"add:{(word[i] + word[i + 1], word[i + 2])}")
                pairs_counts[(word[i] + word[i + 1], word[i + 2])] += freq
            else:
                # increase count of pair that starts with the whole pair
                if verbose:
                    logger.info(
                        f"add:{(word[i] + word[i + 1], word[i + 2] + word[i + 3])}"
                    )
                pairs_counts[(word[i] + word[i + 1], word[i + 2] + word[i + 3])] += freq


class Encoder:
    UNK = "<unk>"
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"

    SPECIAL_TOKENS = [PAD, BOS, EOS]

    def __init__(
        self,
        encoder: Optional[Dict[str, int]] = None,
        bpe_merges: Optional[List[Pair]] = None,
        max_merges: int = 1000,
        handle_errors: str = "replace",
        verbose: bool = False,
    ):
        self.max_merges = max_merges

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.special_tokens = set(self.SPECIAL_TOKENS)
        self.encoder = {k: i for i, k in enumerate(self.SPECIAL_TOKENS + [self.UNK])}
        self.decoder = {}
        self.bpe_ranks = {}
        self.verbose = verbose

        if encoder is not None:
            self.encoder = encoder
            self.initialize_decoding(bpe_merges)

        self.handle_errors = handle_errors  # how to handle errors in decoding

        self.cache = {}  # cache for the bpe function that stores already processed words

    def clean_cache(
        self,
    ):
        self.cache = {}

    def initialize_decoding(self, bpe_merges):
        logger.info("build decoder vocabulary")
        self.decoder = {v: k for k, v in self.encoder.items()}
        logger.info("enumerate bpe merges")
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

    def byte_encode(self, word: str) -> Word:
        """
        byte encode a word
        """
        return tuple(self.byte_encoder[b] for b in (word + " ").encode("utf-8"))

    def bpe(self, initial_word: Word) -> Word:
        """ "
        bpe encode a word
        """
        if initial_word in self.cache:
            return self.cache[initial_word]

        word = initial_word
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            pair = min(pairs, key=lambda x: self.bpe_ranks.get(x, float("inf")))
            if pair not in self.bpe_ranks:
                break
            new_word, _ = find_and_replace(word, pair)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        self.cache[initial_word] = word
        return word

    def fit(self, texts: List[List[str]]):
        """
        fit the encoder and collect bpe_merges from the list of texts.
        texts are split into words already.
        """
        logger.info("Fitting encoder!")
        logger.info("count unique words")
        words_counts = Counter()
        for text in texts:
            words_counts.update(
                self.byte_encode(word) for word in text if word not in self.special_tokens
            )
        logger.info(f"total words: {len(words_counts)}")

        logger.info("count unique pairs of symbols, and collect unique symbols")
        pairs_counts = Counter()
        symbols = set()
        for word, freq in words_counts.items():
            count_pairs(word, pairs_counts, freq)
            symbols.update(set(word))

        logger.info(f"total pairs: {len(pairs_counts)}")
        logger.info("add unique symbols to the encoder vocabulary")
        for s in symbols:
            self.encoder[s] = len(self.encoder)
        logger.info(f"total symbols: {len(symbols)}")

        logger.info("make merges and update vocabulary")
        # stop when there are only pair that occur only once
        #  or when the number of merges is greater than max_merge
        #  or when the number of symbols in the vocabulary is greater than vocab_size
        bpe_merges = []
        reverse_cache = {}
        while len(bpe_merges) < self.max_merges and len(pairs_counts) > 0:
            #  get the most frequent pair
            pair, pair_freq = pairs_counts.most_common(1)[0]
            if pair_freq > 1:
                # add pair to the list of bpe_merges
                bpe_merges.append(pair)
                if self.verbose:
                    logger.info(
                        f"add {pair} to the list of bpe_merges,"
                        f" total merges {len(bpe_merges)} and vocab size {len(self.encoder)}"
                    )
                elif len(bpe_merges) % 100 == 0:
                    logger.info(
                        f"total merges {len(bpe_merges)} and vocab size {len(self.encoder)}"
                    )
                # add pair to the encoder vocabulary
                if pair[0] + pair[1] not in self.encoder:
                    self.encoder[pair[0] + pair[1]] = len(self.encoder)
                # delete pair from the counts of pairs
                pairs_counts.pop(pair)

                # update words and counts of pairs with the new pair replaced
                new_words = {}
                for word, freq in words_counts.items():
                    # get new word and the start positions of the pair
                    new_word, starts = find_and_replace(word, pair, collect_starts=True)
                    # update counts of pairs with the new word
                    update_pairs_counts_from_merge(
                        word, freq, pairs_counts, starts, verbose=self.verbose
                    )

                    # if the word is not the same as the new word, add it to the list of new words
                    if new_word != word:
                        new_words[new_word] = word
                        reverse_cache[new_word] = reverse_cache.get(word, word)

                #  update words counts with the new words
                for new_word, word in new_words.items():
                    words_counts[new_word] = words_counts[word]
                    words_counts.pop(word)

            else:
                break

        logger.info(
            f"merging has finished,"
            f" total merges {len(bpe_merges)} and vocab size {len(self.encoder)}"
        )
        logger.info(f"build cache for fast bpe encoding")
        self.cache = {v: k for k, v in reverse_cache.items()}
        self.initialize_decoding(bpe_merges)

    def encode(self, words: List[str]) -> List[int]:
        bpe_tokens = []
        for word in words:
            if word in self.special_tokens:
                bpe_tokens.append(self.encoder[word])
            else:
                bpe_word = self.bpe(self.byte_encode(word))
                if self.verbose:
                    logger.info(f"word: {word},bpe: {bpe_word}")
                bpe_tokens.extend(
                    self.encoder.get(bpe_token, self.encoder[self.UNK])
                    for bpe_token in bpe_word
                )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> List[str]:
        decoded = [self.decoder[token] for token in tokens]
        text = ""
        for token in decoded:
            if token in self.special_tokens:
                text += token + " "
            else:
                text += bytearray([self.byte_decoder[c] for c in token]).decode(
                    "utf-8", errors=self.handle_errors
                )
        text = text.strip().split(" ")
        return text


def save_encoder(encoder: Encoder, path):
    encoder_file = Path(path, "encoder.json")
    with open(encoder_file, "w") as f:
        json.dump(encoder.encoder, f)

    bpe_file = Path(path, "bpe.txt")
    with open(bpe_file, "w") as f:
        for k, v in sorted(encoder.bpe_ranks.items(), key=lambda x: x[1]):
            f.write(k[0] + " " + k[1] + "\n")


def load_encoder(path):
    encoder_file = Path(path, "encoder.json")
    if not encoder_file.exists():
        raise FileNotFoundError(
            f"Encoder file {encoder_file} does not exist. Please check the path."
        )
    with open(encoder_file, "r") as f:
        encoder = json.load(f)

    bpe_file = Path(path, "bpe.txt")
    if not bpe_file.exists():
        raise FileNotFoundError(
            f"BPE file {bpe_file} does not exist. Please check the path."
        )
    with open(bpe_file, "r") as f:
        bpe_merges = f.read()

    bpe_merges = [
        tuple(line.strip().split(" ")) for line in bpe_merges.split("\n") if line.strip()
    ]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)
