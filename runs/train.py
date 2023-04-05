import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from model import VanillaTransformer, VanillaTransformerConfig


class TrainingModel(pl.LightningModule):
    def __init__(
        self,
        model=VanillaTransformer,
        config=VanillaTransformerConfig(),
        training_config=None,
    ):
        super().__init__()
        self.model = model(config)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.metric = Accuracy(task="multiclass", num_classes=config.output_vocab_size)
        self.automatic_optimization = False
        self.training_config = training_config

    def forward(self, x, targets, encoder_mask=None, decoder_mask=None):
        return self.model(x, targets, encoder_mask, decoder_mask)

    def training_step(self, batch, batch_idx):
        x, targets, encoder_mask, tgt_len = batch
        n = torch.randint(1, torch.max(tgt_len).item(), size=(1,)).item()
        to_predict = targets[:, 1 : n + 1].long()
        targets = targets[:, :n]
        logits, _ = self(x, targets, encoder_mask)
        loss = self.loss(logits[:, -1, :].view(-1, logits.size(-1)), to_predict[:, -1])

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(
            opt,
            gradient_clip_val=self.training_config.gradient_clip_val,
            gradient_clip_algorithm=self.training_config.gradient_clip_algorithm,
        )
        opt.step()
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets, encoder_mask, tgt_len = batch
        result = self.model.generate(
            x,
            encoder_mask=encoder_mask,
            max_len=targets.size(1),
            do_sample=True,
            sos_token_id=targets[0, 0],
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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            eps=self.training_config.eps,
        )
