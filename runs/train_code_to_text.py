import pytorch_lightning as pl
import torch

from model import VanillaTransformerConfig
from runs.train import TrainingModel


class TrainingConfig:
    lr = 1e-3
    betas = (0.9, 0.98)
    eps = 1e-9
    gradient_clip_val = 1.0
    gradient_clip_algorithm = "norm"


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.code_to_text import CodeToText

    train_dataset = CodeToText("train", "../dataset/bpe_encoder")
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

    validation_dataset = CodeToText("validation", "../dataset/bpe_encoder")
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    config_ = VanillaTransformerConfig(
        input_vocab_size=train_dataset.get_input_vocab_size(),
        output_vocab_size=train_dataset.get_output_vocab_size(),
        d_seq=train_dataset.max_length,
        d_embed=64,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
    )

    model_ = TrainingModel(config=config_, training_config=TrainingConfig)
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=int(torch.cuda.is_available()),
    )

    trainer.fit(model=model_, train_dataloaders=loader, val_dataloaders=validation_loader)
