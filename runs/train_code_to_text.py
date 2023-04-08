import pytorch_lightning as pl
import torch

from runs.train_vanilla import TrainingModel
from tmodels import VanillaTransformerConfig


class TrainingConfig:
    lr = 1e-3
    betas = (0.9, 0.98)
    eps = 1e-9
    gradient_clip_val = 1.0
    gradient_clip_algorithm = "norm"


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from runs.data import CodeToText

    train_dataset = CodeToText("train", "dataset/bpe_encoder", max_length=128)
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)

    validation_dataset = CodeToText("validation", "dataset/bpe_encoder", max_length=128)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    config_ = VanillaTransformerConfig(
        input_vocab_size=train_dataset.get_input_vocab_size(),
        output_vocab_size=train_dataset.get_output_vocab_size(),
        d_seq=train_dataset.max_length,
        d_embed=64,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
    )

    # model_ = TrainingModel(config=config_, training_config=TrainingConfig)
    model_ = TrainingModel._load_from_checkpoint(
        "lightning_logs/version_24/checkpoints/epoch=0-step=7870.ckpt"
    )
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=int(torch.cuda.is_available()),
    )

    trainer.fit(model=model_, train_dataloaders=loader, val_dataloaders=validation_loader)
