import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class BoardScoreGenerator(pl.LightningModule):
    def __init__(
        self,
        num_tiles=13, # King, knight, bishop, queen, rook, pawn, 2 colors, empty cell
        embedding_dim=64,
        conv_channels=512,
        ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels
        self.num_tiles = num_tiles  # Could be interesting to explore a positional embedding too

        self.save_hyperparameters()

        self.embedding = torch.nn.Embedding(num_tiles, embedding_dim)
        self.nn1 = torch.nn.Conv2d(embedding_dim, self.conv_channels, 3, padding=1)
        self.nn2 = torch.nn.Conv2d(self.conv_channels, self.conv_channels, 3)
        self.ff0 = torch.nn.Linear(self.conv_channels*36,self.conv_channels)
        self.ff1 = torch.nn.Linear(self.conv_channels,1)

    def forward(self, x):
        # x is [B, 8, 8] type long
        x = self.embedding(x)
        # [B, 8, 8, emb_dim]
        x = torch.transpose(x,1,3)
        # [B, emb_dim, 8, 8]
        x = self.nn1(x)
        x = F.relu(x)
        # [B, emb_dim, 8, 8]
        x = self.nn2(x)
        x = F.relu(x)
        # [B, emb_dim, 6, 6]
        x = x.flatten(start_dim=1)
        # [B, emb_dim*36]
        x = self.ff0(x)
        # [B, emb_dim]
        x = self.ff1(x)
        # [B, 1]
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.cuda(0)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log('train_mse_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.cuda(0)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log('val_mse_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
        lr_fn = lambda epoch: [0.1, 0.1, 100][epoch % 3]
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_fn)
        return [optimizer], [scheduler]
