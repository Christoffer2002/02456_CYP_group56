

def unfreeze_encoder(model, optimizer, epoch, unfreeze_epoch):
    """
    Unfreezes the encoder of the model after a specified epoch.

    """
    if epoch == unfreeze_epoch:
        model.cfg.freeze_encoder = False
        model.encoder.requires_grad_(True)
        model.encoder.set_grad_checkpointing(True)

        optimizer.add_param_group({'params': model.encoder.parameters(), 'lr': model.cfg.encoder_lr})

