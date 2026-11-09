"""Offline CPU tests, no pretrained downloads and no real research results."""
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gradient_weighting import (ren_head_gate, head_gradients, head_features,
                           adam_directions, reserve_meta)


def check_math():
    torch.manual_seed(7)
    h, mh = torch.randn(6, 5), torch.randn(9, 5)
    y, my = torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 2] * 3)
    layer = torch.nn.Linear(5, 3)
    logits = layer(h)
    features = head_gradients(h, logits.detach(), y)
    for i in range(len(y)):
        loss = F.cross_entropy(layer(h[i:i+1]), y[i:i+1], label_smoothing=0.1)
        grad = torch.autograd.grad(loss, (layer.weight, layer.bias))
        torch.testing.assert_close(features[i], torch.cat([g.flatten() for g in grad]))
    before = copy.deepcopy(layer.state_dict())
    gate, sensitivity, fallback = ren_head_gate(h, y, mh, my, layer.weight,
                                               layer.bias, 0.2)
    meta_loss = F.cross_entropy(layer(mh), my, label_smoothing=0.1)
    mg = torch.cat([g.flatten() for g in torch.autograd.grad(meta_loss,
                                                          (layer.weight, layer.bias))])
    # Exact derivative at eps=0: alpha/n * <g_i, g_meta>.
    torch.testing.assert_close(sensitivity, 0.2 / len(y) * (features @ mg),
                               atol=2e-7, rtol=1e-5)
    assert not gate.requires_grad and float(gate.min()) >= 0
    assert abs(float(gate.mean()) - 1) < 1e-5
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(p.grad is None for p in layer.parameters())
    zero_w, zero_b = torch.zeros(3, 2), torch.zeros(3)
    gate, _, fallback = ren_head_gate(torch.ones(2, 2), torch.ones(2, dtype=torch.long),
        torch.ones(1, 2), torch.zeros(1, dtype=torch.long), zero_w, zero_b, 0.1)
    assert fallback and torch.equal(gate, torch.ones(2))
    model = SimpleNamespace(classifier=layer)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    for p in layer.parameters():
        optimizer.state[p] = {'exp_avg': torch.full_like(p, 0.2),
                              'exp_avg_sq': torch.full_like(p, 0.3),
                              'step': torch.tensor(4.)}
    state = copy.deepcopy(optimizer.state_dict())
    result, step = adam_directions(features, model, optimizer)
    expected = (0.9 * 0.2 + 0.1 * features) / torch.sqrt(
               0.999 * 0.3 + 0.001 * features.square() + 1e-8)
    torch.testing.assert_close(result, expected)
    assert step == 4
    for key, value in optimizer.state_dict()['state'].items():
        for field in value:
            torch.testing.assert_close(value[field], state['state'][key][field])
    print('PASS math: exact classifier gradients, Ren meta-derivative, fallback, LESS moments.')


def check_runner():
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (DistilBertConfig, DistilBertForSequenceClassification,
        PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorWithPadding)
    import run_experiments as runner
    from data_utils import LABELS
    from evaluation import compute_metrics

    def frame(prefix, per_class):
        return pd.DataFrame([{'sentence': f'{prefix} class{c} item{i}', 'label': c}
                              for c in range(3) for i in range(per_class)])
    frames = {'seed': frame('seed', 2), 'validation': frame('validation', 5),
              'test': frame('test', 2), 'train_pool': frame('pool', 9)}
    synthetic = frame('synthetic', 3)
    split = reserve_meta(frames, synthetic, 1, 24266)
    assert len(split['meta']) == 3 and len(split['validation']) == 12
    assert split['meta'].equals(reserve_meta(frames, synthetic, 1, 24266)['meta'])
    assert set(split['meta'].sentence).isdisjoint(split['validation'].sentence)
    backend = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'class0': 2,
                                  'class1': 3, 'class2': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                  unk_token='[UNK]', model_input_names=['input_ids', 'attention_mask'])

    def make_seed(args, output, data):
        cfg = DistilBertConfig(vocab_size=5, n_layers=1, dim=12, hidden_dim=24,
                  n_heads=2, num_labels=3, id2label=LABELS['phrasebank'],
                  label2id={v: k for k, v in LABELS['phrasebank'].items()})
        model = DistilBertForSequenceClassification(cfg)
        def tokens(frame):
            return Dataset.from_pandas(frame[['sentence', 'label']], preserve_index=False).map(
                lambda b: tokenizer(b['sentence'], truncation=True, max_length=32),
                batched=True, remove_columns=['sentence'])
        train_args = TrainingArguments(output_dir=str(output / 'checkpoints'),
            eval_strategy='epoch', save_strategy='no', num_train_epochs=2,
            per_device_train_batch_size=16, per_device_eval_batch_size=32,
            learning_rate=1e-4, report_to='none', seed=args.train_seed,
            data_seed=args.train_seed, dataloader_num_workers=0, disable_tqdm=True)
        trainer = Trainer(model=model, args=train_args, processing_class=tokenizer,
            train_dataset=tokens(data['seed']), eval_dataset=tokens(data['validation']),
            data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
        # Test scoring neither alters gradients/mode nor consumes dropout RNG.
        model.train()
        rng = torch.get_rng_state().clone()
        original = copy.deepcopy(model.state_dict())
        head_features(model, tokenizer, data['meta'], 2, 32)
        assert model.training and torch.equal(rng, torch.get_rng_state())
        assert all(p.grad is None for p in model.parameters())
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, original[k])
        return trainer, tokenizer, None, None, None

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, sentences, **kwargs):
            result = []
            for s in sentences:
                seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
                vector = np.random.default_rng(seed).normal(size=8)
                result.append(vector / np.linalg.norm(vector))
            return np.array(result)

    old_argv = sys.argv[:]
    old_module = sys.modules.get('sentence_transformers')
    originals = (runner.prepare_data, runner.prepare_synthetic,
                 runner.build_seed_trainer, runner.assigned_label_probabilities)
    try:
        runner.prepare_data = lambda **kw: (copy.deepcopy(frames), {'fixture': True})
        runner.prepare_synthetic = lambda *a: (synthetic.copy(), [])
        runner.build_seed_trainer = make_seed
        runner.assigned_label_probabilities = lambda *a: np.full(len(synthetic), 0.8)
        sys.modules['sentence_transformers'] = SimpleNamespace(SentenceTransformer=FakeEncoder)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results'
            sys.argv = ['run_experiments.py', '--datasets', 'phrasebank',
                        '--meta-per-class', '1', '--num-epochs', '2', '--patience', '2',
                        '--output-root', str(output)]
            runner.main()
            comparison = pd.read_csv(output / 'comparison.csv')
            assert set(comparison.experiment) == set(runner.EXPERIMENTS)
            assert len(comparison) == 7
            assert not list(output.rglob('*.safetensors'))
            assert not list(output.rglob('optimizer.pt'))
            base = output / 'phrasebank/train_seed_42'
            reference = None
            for e in runner.EXPERIMENTS:
                predictions = pd.read_csv(base / e / 'test_predictions.csv')
                current = predictions[['sentence', 'label']]
                assert reference is None or reference.equals(current)
                reference = current
            for e in ['E5', 'E7']:
                history = pd.read_csv(base / e / 'ren_batch_history.csv')
                weights = pd.read_csv(base / e / 'ren_sample_weights.csv')
                assert len(history) == 2  # no extra rows from validation or test
                assert weights.weight.between(0.5, 2).all()
                assert len(weights) == 2 * len(synthetic)
            assert len(list((base / 'E3/weights').glob('*.csv'))) == 1
            for e in ['E4', 'E5', 'E6', 'E7']:
                assert len(list((base / e / 'weights').glob('*.csv'))) == 2
            assert pd.read_csv(base / 'E6/weight_history.csv').adam_moment_step.iloc[1] > 0
            warmup = json.loads((base / 'warmup_moment_info.json').read_text())
            e1_selection = json.loads((base / 'E1/selection.json').read_text())
            assert warmup['source_epoch'] == e1_selection['best_epoch']
            for e in ['E6', 'E7']:
                weights = pd.read_csv(base / e / 'weight_history.csv')
                assert weights.adam_moment_source.tolist() == ['M0_best', 'current_branch']
                assert weights.adam_moment_step.iloc[0] == warmup['optimizer_step']
            print('PASS runner: all E1--E7, 2 epochs, real HF training, predictions, RAM selection.')
            # Probe the validation-only path and ensure it cannot call test_and_save.
            previous_test = runner.test_and_save
            def forbidden_test(*args, **kwargs):
                raise AssertionError('Validation-only must not evaluate test.')
            runner.test_and_save = forbidden_test
            try:
                val_output = Path(temporary) / 'validation_only'
                sys.argv = ['run_experiments.py', '--datasets', 'phrasebank',
                    '--experiments', 'E7', '--meta-per-class', '1', '--num-epochs', '1',
                    '--validation-only', '--output-root', str(val_output)]
                runner.main()
                assert not list(val_output.rglob('test_predictions.csv'))
                val_table = pd.read_csv(val_output / 'comparison.csv')
                assert set(val_table.experiment) == {'E1', 'E7'}
                assert 'test_macro_f1' not in val_table.columns
            finally:
                runner.test_and_save = previous_test
            print('PASS validation-only: no test evaluation, still auto-trains E1/M0.')
    finally:
        sys.argv = old_argv
        (runner.prepare_data, runner.prepare_synthetic, runner.build_seed_trainer,
         runner.assigned_label_probabilities) = originals
        if old_module is None:
            sys.modules.pop('sentence_transformers', None)
        else:
            sys.modules['sentence_transformers'] = old_module


if __name__ == '__main__':
    torch.set_num_threads(1)
    check_math()
    check_runner()
    print('ALL OFFLINE CHECKS PASSED. Tiny CPU fixtures are not research results.')
