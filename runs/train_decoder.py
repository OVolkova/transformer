import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from tmodels import DecoderOnly, DecoderOnlyConfig


class TrainingDecoder(pl.LightningModule):
    def __init__(
        self,
        model=DecoderOnly,
        config=DecoderOnlyConfig(),
        training_config=None,
    ):
        super().__init__()
        self.last_token = config.vocab_size - 1
        self.model = model(config)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.metric = Accuracy(
            task="multiclass", num_classes=config.vocab_size, ignore_index=-1
        )
        self.automatic_optimization = False
        self.training_config = training_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        logits = self(x)
        loss = self.loss(logits.view(-1, logits.size(-1)), y.view(-1))

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
        x, y, q_len, a_len = batch
        x = x[:, : q_len.max()]
        result = self.model.generate(
            x, max_len=y.size(1), do_sample=True, last_token=self.last_token
        )
        if y.size(1) > result.size(1):
            result = torch.cat(
                (
                    result,
                    torch.zeros((y.size(0), y.size(1) - result.size(1))).long(),
                ),
                dim=1,
            )
        else:
            result = result[:, : y.size(1)]
        self.metric.update(result.contiguous().view(-1), y.contiguous().view(-1))

    def validation_epoch_end(self, outputs):
        value = self.metric.compute()
        print(value)
        self.log("valid_accuracy", value)
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            eps=self.training_config.eps,
        )
