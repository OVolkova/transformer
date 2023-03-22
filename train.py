import pytorch_lightning as pl
import torch
from model import VanillaTransformer, VanillaTransformerConfig
from torchmetrics import Accuracy


class TrainingModel(pl.LightningModule):
    def __init__(self, model=VanillaTransformer, config=VanillaTransformerConfig()):
        super().__init__()
        self.model = model(config)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metric = Accuracy(task="multiclass", num_classes=config.output_vocab_size)

    def forward(self, x, targets, encoder_mask=None, decoder_mask=None):
        return self.model(x, targets, encoder_mask, decoder_mask)

    def training_step(self, batch, batch_idx):
        x, targets, encoder_mask, tgt_len = batch
        n = torch.randint(1, torch.max(tgt_len).item(), size=(1,)).item()
        to_predict = targets[:, 1: n + 1].long()
        targets = targets[:, :n]
        logits, _ = self(x, targets, encoder_mask)
        loss = self.loss(
            logits[:,-1,:].view(-1, logits.size(-1)), to_predict[:,-1]
        )
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets, encoder_mask, tgt_len = batch
        result = self.model.generate(
            x, encoder_mask=encoder_mask, max_len=10, do_sample=True, sos_token_id=25
        )
        if targets.size(1) > result.size(1):
            result = torch.cat(
                (
                    result,
                    torch.zeros(
                        (targets.size(0), targets.size(1) - result.size(1))
                    ).long(),
                ),
                dim=1,
            )
        else:
            result = result[:, : targets.size(1)]
        value = self.metric(result.contiguous().view(-1), targets.contiguous().view(-1))
        self.log("test accuracy", value)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from data.demo import NumbersRule
    from torch.utils.data import DataLoader

    train_dataset = NumbersRule("train")
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    test_dataset = NumbersRule("test")
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    config_ = VanillaTransformerConfig(
        input_vocab_size=train_dataset.get_input_vocab_size(),
        output_vocab_size=train_dataset.get_output_vocab_size(),
        d_seq=train_dataset.length+2,
    )

    model_ = TrainingModel(config=config_)
    # model_ = TrainingModel._load_from_checkpoint(
    #     "/home/olly/Documents/projects/transformer/transformer/lightning_logs/version_90/checkpoints/epoch=499-step=312500.ckpt"
    # )
    trainer = pl.Trainer(max_epochs=30, gpus=int(torch.cuda.is_available()), profiler="simple")

    trainer.fit(model=model_, train_dataloaders=loader, val_dataloaders=test_loader)
    # trainer.test(dataloaders=test_loader)
