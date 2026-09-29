from transformers import TrainerCallback


class LogExportCallback(TrainerCallback):
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        epoch = logs.get("epoch")

        if epoch is None:
            return

        # Training loss
        if "loss" in logs:
            self.history["train_loss"].append(logs["loss"])

        # Validation loss
        if "eval_loss" in logs:
            self.history["val_loss"].append(logs["eval_loss"])

        # Validation accuracy
        if "eval_accuracy" in logs:
            self.history["val_accuracy"].append(logs["eval_accuracy"])

        # Record epoch once per validation/training logging cycle.
        # The plotting code expects one epoch per validation result.
        if "eval_loss" in logs:
            self.history["epoch"].append(epoch)
