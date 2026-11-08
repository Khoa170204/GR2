"""Offline checks: no dataset/model download, no training checkpoint output."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, BatchEncoding
from gradient_weights import linear_ce_features, score_to_weights, split_meta, GradientWeights
from dynamic_weights import MutableWeights
from data_utils import LABELS


def main():
    torch.manual_seed(42)
    head = torch.nn.Linear(7, 3)
    hidden = torch.randn(5, 7)
    labels = torch.tensor([0, 1, 2, 0, 1])
    for smoothing in [0.0, 0.1]:
        logits = head(hidden)
        actual = linear_ce_features(hidden, logits, labels, smoothing)
        for i in range(5):
            loss = torch.nn.functional.cross_entropy(
                head(hidden[i:i+1]), labels[i:i+1], label_smoothing=smoothing)
            gradients = torch.autograd.grad(loss, (head.weight, head.bias))
            expected = torch.cat([g.flatten() for g in gradients])
            torch.testing.assert_close(actual[i], expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(score_to_weights([-1, -0.4, 0, 0.6, 1]), [.5, .8, 1, 1.6, 2])
    validation = pd.DataFrame([{'sentence': f'val {c} {i}', 'label': c}
                               for c in range(3) for i in range(5)])
    synthetic = pd.DataFrame({'sentence': ['syn a', 'syn b', 'syn c'], 'label': [0, 1, 2]})
    frames = {'seed': pd.DataFrame({'sentence': ['seed'], 'label': [0]}),
              'validation': validation,
              'test': pd.DataFrame({'sentence': ['test'], 'label': [0]})}
    split = split_meta(frames, synthetic, 2, 42)
    assert len(split['meta']) == 6 and len(split['validation']) == 9
    assert not set(split['meta'].sentence) & set(split['validation'].sentence)
    assert split['meta'].equals(split_meta(frames, synthetic, 2, 42)['meta'])

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = [[1, 2 + sum(map(ord, text)) % 26, 2] for text in texts]
            tensor = torch.tensor(ids)
            return BatchEncoding({'input_ids': tensor, 'attention_mask': torch.ones_like(tensor)})

    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=32, n_layers=1, n_heads=2, dim=12, hidden_dim=24,
        num_labels=3, id2label=LABELS['phrasebank']))
    args = SimpleNamespace(world_size=1, dataloader_num_workers=0)
    with tempfile.TemporaryDirectory() as directory:
        for mode, expected_files in [('static', 1), ('dynamic', 2)]:
            folder = Path(directory) / mode
            folder.mkdir()
            data = MutableWeights([{} for _ in range(4)], 1, 3)
            updater = GradientWeights(data, synthetic, split['meta'], Tokenizer(),
                                      'phrasebank', mode, folder, batch_size=2, max_length=8)
            model.train()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            for epoch in [0, 1]:
                rng = torch.random.get_rng_state().clone()
                updater.on_epoch_begin(args, SimpleNamespace(epoch=epoch), object(), model=model)
                assert model.training
                assert torch.equal(rng, torch.random.get_rng_state())
            assert data.weights[0] == 1.0
            assert np.all((data.weights >= .5) & (data.weights <= 2))
            assert len(list((folder/'weights').glob('epoch_*.csv'))) == expected_files
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, before[k])
            assert all(p.grad is None for p in model.parameters())
    print('PASS: analytic gradients, mapping, meta split, callbacks, model/RNG preservation.')


if __name__ == '__main__':
    main()
