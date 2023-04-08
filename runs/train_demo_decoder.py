import pytorch_lightning as pl
import torch

from runs.train_decoder import TrainingDecoder
from tmodels import DecoderOnlyConfig

# from pytorch_lightning.loggers import WandbLogger


class TrainingConfig:
    lr = 1e-3
    betas = (0.9, 0.98)
    eps = 1e-9
    gradient_clip_val = 1.0
    gradient_clip_algorithm = "norm"


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from runs.data.demo_decoder_only import NumbersRule

    train_dataset = NumbersRule("train", length=10000 * 4)
    loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    test_dataset = NumbersRule("test", length=50)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    config_ = DecoderOnlyConfig(
        vocab_size=train_dataset.get_vocab_size(),
        d_seq=train_dataset.max_length,
    )

    model_ = TrainingDecoder(config=config_, training_config=TrainingConfig)
    # model_ = TrainingModel._load_from_checkpoint(
    #     "lightning_logs/version_90/checkpoints/epoch=499-step=312500.ckpt"
    # )
    # wandb_logger = WandbLogger(name='Adam-32-0.001', project='transformers')
    trainer = pl.Trainer(
        max_epochs=500,
        gpus=int(torch.cuda.is_available()),
        # logger=wandb_logger
    )

    trainer.fit(model=model_, train_dataloaders=loader, val_dataloaders=test_loader)
    # trainer.test(dataloaders=test_loader)
