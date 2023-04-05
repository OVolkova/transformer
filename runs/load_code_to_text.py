import pandas as pd
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    for split in ["train", "test", "validation"]:
        df = pd.DataFrame(
            zip(dataset[split]["code_tokens"], dataset[split]["docstring_tokens"]),
            columns=["code_tokens", "docstring_tokens"],
        )
        df.to_parquet("../dataset/code_to_text/" + split + ".parquet")
