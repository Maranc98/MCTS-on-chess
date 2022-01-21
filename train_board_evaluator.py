if __name__ == '__main__':
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, random_split
    from models import BoardEvaluationsDataset, BoardScoreGenerator

    BATCH_SIZE = 2048
    NUM_WORKERS = 0

    dataset = BoardEvaluationsDataset('./datasets/chessData.csv')
    train_len = int(len(dataset)*0.7)
    tr, vl = random_split(dataset, [train_len, len(dataset)-train_len])

    train_dataloader = DataLoader(tr, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_dataloader = DataLoader(vl, batch_size=BATCH_SIZE)

    model = BoardScoreGenerator(embedding_dim=64,conv_channels=256)

    logger = pl.loggers.tensorboard.TensorBoardLogger('./tb_logs')

    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val_loss"))
    callbacks.append(pl.callbacks.LearningRateMonitor(log_momentum=True))

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
