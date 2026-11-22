"""Train the shared seed model; keep only its metrics and fingerprint on disk."""
import hashlib
from types import SimpleNamespace

from evaluation import fit_in_ram
from run_io import write_json
from scoring import release_gpu
from training import build_seed_trainer


def train_m0(config, frames, folder):
    print(f'[M0] Training on {len(frames["seed"])} seed rows: {config["dataset"]}', flush=True)
    trainer, tokenizer, *_ = build_seed_trainer(SimpleNamespace(**config), folder, frames)
    selection = fit_in_ram(trainer, config['early_stopping_patience'])
    model = trainer.model.cpu()
    write_json(folder / 'm0_training_history.json', trainer.state.log_history)
    del trainer
    release_gpu()

    # Identify this initialization across runs without serializing the parameters.
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(memoryview(tensor.detach().contiguous().numpy()).cast('B'))
    return model, tokenizer, selection, digest.hexdigest()
