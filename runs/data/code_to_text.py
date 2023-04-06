from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tmodels.bpe import Encoder
from tmodels.bpe.bpe import load_encoder, save_encoder
from tmodels.logger import logger


# TODO proper file paths
def load_data(split="train") -> Tuple[np.array, List[List[str]]]:
    dataset = pd.read_parquet("../dataset/code_to_text/" + split + ".parquet")
    return dataset["code_tokens"].values, dataset["docstring_tokens"].values


class CodeToText(Dataset):
    """
    input: code tokens
    output: bos + docstring tokens + eos

    """

    def __init__(self, split: str, bpe_path: str, max_length=32):
        assert split in {"train", "test", "validation"}
        self.max_length = max_length
        self.split = split
        source, target = load_data(split=split)
        self.input, self.in_padding_token, self.in_vocab_size = self.read_and_encode(
            "code_tokens", bpe_path, source, is_input=True
        )
        self.output, self.out_padding_token, self.out_vocab_size = self.read_and_encode(
            "docstring_tokens", bpe_path, target, is_input=False
        )

    def read_and_encode(
        self, column: str, bpe_path: str, data: List[List[str]], is_input: bool = False
    ):
        encoder = self.read_encoder(bpe_path, column, data)

        logger.info(f"Encoding data for {column}")
        data = [
            encoder.encode(line if is_input else [encoder.BOS] + line + [encoder.EOS])
            for line in data
        ]
        padding_token = encoder.encode([encoder.PAD])[0]
        return data, padding_token, len(encoder.encoder)

    def read_encoder(self, bpe_path: str, column: str, data: List[List[str]]):
        try:
            logger.info(f"Loading encoder for {column}")
            encoder = load_encoder(Path(bpe_path, column))
        except FileNotFoundError as error:
            if self.split == "train":
                logger.info(f"encoder for {column} not found, creating new encoder")
                encoder = Encoder()
                encoder.fit(data)
                save_encoder(encoder, Path(bpe_path, column))
            else:
                logger.info(f"encoder for {column} not found, failing...")
                raise FileNotFoundError(error)
        # encoder.verbose = True
        return encoder

    def __len__(self):
        return len(self.input)

    def get_input_vocab_size(self):
        return self.in_vocab_size

    def get_output_vocab_size(self):
        return self.out_vocab_size

    def get_block_size(self):
        return self.max_length

    def __getitem__(self, idx):
        source = torch.tensor(self.input[idx], dtype=torch.long)
        target = torch.tensor(self.output[idx], dtype=torch.long)
        target_len = min(self.max_length, target.size(0))
        if len(source) > self.max_length:
            source = source[: self.max_length]
        else:
            source = torch.cat(
                (
                    source,
                    torch.tensor(
                        [self.in_padding_token] * (self.max_length - len(source)),
                        dtype=torch.long,
                    ),
                )
            )
        padding_mask = torch.cat(
            (
                torch.ones(len(source), dtype=torch.bool),
                torch.zeros(self.max_length - len(source), dtype=torch.bool),
            )
        )
        if len(target) > self.max_length:
            target = target[: self.max_length]
        else:
            target = torch.cat(
                (
                    target,
                    torch.tensor(
                        [self.out_padding_token] * (self.max_length - len(target)),
                        dtype=torch.long,
                    ),
                )
            )
        return source, target, padding_mask, target_len
