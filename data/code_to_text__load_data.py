from datasets import load_dataset
import os
import tqdm
from pathlib import Path


def save_data(dataset, column, path, split='train'):
    if not Path(path).is_dir():
        os.mkdir(path)
    path = Path(path, split)
    if not path.is_dir():
        os.mkdir(path)
    file_path = Path(path, column + ".txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            f.write(' '.join(dataset[split][i][column]).replace("\n", ' ') + "\n")


if __name__ == '__main__':
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    data_path = 'dataset/code_to_text'
    split = 'train'
    column = 'code_tokens'
    save_data(dataset, column, data_path, split=split)
    column = 'docstring_tokens'
    save_data(dataset, column, data_path, split=split)

    split = 'validation'
    column = 'code_tokens'
    save_data(dataset, column, data_path, split=split)
    column = 'docstring_tokens'
    save_data(dataset, column, data_path, split=split)

    split = 'test'
    column = 'code_tokens'
    save_data(dataset, column, data_path, split=split)
    column = 'docstring_tokens'
    save_data(dataset, column, data_path, split=split)
