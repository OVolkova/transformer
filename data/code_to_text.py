from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from logger import logger
from model.bpe import Encoder, load_encoder, save_encoder


def load_data(column, path, split="runs"):
    file_path = Path(Path(path, split), column + ".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [line.strip().split(" ") for line in f.readlines()][:-1]
    return data


class CodeToText(Dataset):
    """
    input: code tokens
    output: bos + docstring tokens + eos

    """

    def __init__(self, split: str, dataset_path: str, bpe_path: str, max_length=512):
        assert split in {"train", "test", "validation"}
        self.max_length = max_length
        self.split = split
        self.input, self.in_encoder = self.read_and_encode(
            "code_tokens", dataset_path, bpe_path, is_input=True
        )
        self.in_padding_token = self.in_encoder.encode([self.in_encoder.PAD])[0]
        self.in_encoder.clean_cache()
        self.output, self.out_encoder = self.read_and_encode(
            "docstring_tokens", dataset_path, bpe_path, is_input=False
        )
        self.out_padding_token = self.out_encoder.encode([self.out_encoder.PAD])[0]
        self.out_encoder.clean_cache()

    def read_and_encode(
        self, column: str, dataset_path: str, bpe_path: str, is_input: bool = False
    ):
        logger.info(f"Loading data for {column}")
        data = load_data(column, dataset_path, split=self.split)

        encoder = self.read_encoder(bpe_path, column, data)

        logger.info(f"Encoding data for {column}")
        data = [
            encoder.encode(line if is_input else [encoder.BOS] + line + [encoder.EOS])
            for line in data
        ]
        return data, encoder

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
        return len(self.in_encoder.encoder)

    def get_output_vocab_size(self):
        return len(self.out_encoder.encoder)

    def get_block_size(self):
        return self.max_length

    def __getitem__(self, idx):
        source = torch.tensor(self.input[idx], dtype=torch.long)
        target = torch.tensor(self.output[idx], dtype=torch.long)
        target_len = target.size(0)
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
