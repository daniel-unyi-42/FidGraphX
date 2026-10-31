import os
import torch
from src.utils import apply_mask
from src.metrics import (
    fid_plus_prob, fid_minus_prob,
    fid_plus_acc, fid_minus_acc,
    fid_plus_cat_tv, fid_minus_cat_tv,
    fid_minus_reg_tv, fid_plus_reg_tv,
    precision_score, recall_score, f1_score,
    roc_auc_score, pr_auc_score
)


class ProbeEvaluator:
    def __init__(
        self,
        mask_provider,
        target_model,
        probe_factory,
        task_type,
        target_type,
        log_path,
        checkpoint_tag,
    ):
        self.mask_provider = mask_provider
        self.target_model = target_model
        self.probe_factory = probe_factory
        self.task_type = task_type
        self.target_type = target_type
        self.log_path = log_path
        self.checkpoint_tag = checkpoint_tag
        self.device = target_model.device
        self.eval_pos_probe = None
        self.eval_neg_probe = None

    def _empty_metrics(self):
        return {
            'sparsity': 0.0,
            'pos_loss': 0.0,
            'pos_metric': 0.0,
            'neg_loss': 0.0,
            'neg_metric': 0.0,
            'fidplus_prob': 0.0,
            'fidminus_prob': 0.0,
            'fidplus_acc': 0.0,
            'fidminus_acc': 0.0,
            'fidplus': 0.0,
            'fidminus': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0,
            'roc_auc': 0.0,
        }

    def _build_target(self, baseline_logits):
        if self.task_type == 'classification' and self.target_type == 'soft':
            return baseline_logits.softmax(dim=1)
        if self.task_type == 'classification' and self.target_type == 'hard':
            return baseline_logits.argmax(dim=1)
        return baseline_logits

    def _checkpoint_paths(self):
        pos_path = os.path.join(self.log_path, f'eval_pos_probe_{self.checkpoint_tag}.pt')
        neg_path = os.path.join(self.log_path, f'eval_neg_probe_{self.checkpoint_tag}.pt')
        return pos_path, neg_path

    def initialize_probes(self):
        self.eval_pos_probe = self.probe_factory()
        self.eval_neg_probe = self.probe_factory()

    def train_epoch(self, loader):
        if self.eval_pos_probe is None or self.eval_neg_probe is None:
            raise RuntimeError('Call initialize_probes() before train_epoch().')
        self.target_model.eval()
        metrics = self._empty_metrics()
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                baseline_logits = self.target_model(data)
                target = self._build_target(baseline_logits)
                probs = self.mask_provider.get_probs(data)
                mask = self.mask_provider.sample_mask(data, mode='train', probs=probs)

            self.eval_pos_probe.train()
            self.eval_pos_probe.optimizer.zero_grad()
            pos_logits = self.eval_pos_probe(apply_mask(data, mask))
            pos_loss = self.eval_pos_probe.criterion(pos_logits, target, reduction='none')
            pos_loss.mean().backward()
            self.eval_pos_probe.optimizer.step()

            self.eval_neg_probe.train()
            self.eval_neg_probe.optimizer.zero_grad()
            neg_logits = self.eval_neg_probe(apply_mask(data, 1.0 - mask))
            neg_loss = self.eval_neg_probe.criterion(neg_logits, target, reduction='none')
            neg_loss.mean().backward()
            self.eval_neg_probe.optimizer.step()

            metrics['sparsity'] += mask.mean().item()
            metrics['pos_loss'] += pos_loss.mean().item()
            metrics['neg_loss'] += neg_loss.mean().item()
            metric_target = target.argmax(dim=1) if self.task_type == 'classification' and self.target_type == 'soft' else target
            metrics['pos_metric'] += self.eval_pos_probe.metric(pos_logits, metric_target).item()
            metrics['neg_metric'] += self.eval_neg_probe.metric(neg_logits, metric_target).item()

            if self.task_type == 'classification':
                metrics['fidplus_prob'] += fid_plus_prob(neg_logits, baseline_logits)
                metrics['fidminus_prob'] += fid_minus_prob(pos_logits, baseline_logits)
                metrics['fidplus_acc'] += fid_plus_acc(neg_logits, baseline_logits)
                metrics['fidminus_acc'] += fid_minus_acc(pos_logits, baseline_logits)
                metrics['fidplus'] += fid_plus_cat_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_cat_tv(pos_logits, baseline_logits)
            else:
                metrics['fidplus'] += fid_plus_reg_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_reg_tv(pos_logits, baseline_logits)

            if hasattr(data, 'true'):
                metrics['precision'] += precision_score(mask, data.true)
                metrics['recall'] += recall_score(mask, data.true)
                metrics['f1'] += f1_score(mask, data.true)
                metrics['pr_auc'] += pr_auc_score(probs, data.true)
                metrics['roc_auc'] += roc_auc_score(probs, data.true)

        for metric_name in metrics:
            metrics[metric_name] /= len(loader)
        return metrics

    @torch.no_grad()
    def evaluate_epoch(self, loader):
        if self.eval_pos_probe is None or self.eval_neg_probe is None:
            raise RuntimeError('Call initialize_probes() before evaluate_epoch().')
        self.target_model.eval()
        self.eval_pos_probe.eval()
        self.eval_neg_probe.eval()
        metrics = self._empty_metrics()

        for data in loader:
            data = data.to(self.device)
            baseline_logits = self.target_model(data)
            target = self._build_target(baseline_logits)
            probs = self.mask_provider.get_probs(data)
            mask = self.mask_provider.sample_mask(data, mode='eval', probs=probs)
            if mask.sum() == 0:
                mask[0] += 1

            pos_logits = self.eval_pos_probe(apply_mask(data, mask))
            neg_logits = self.eval_neg_probe(apply_mask(data, 1.0 - mask))
            pos_loss = self.eval_pos_probe.criterion(pos_logits, target, reduction='none')
            neg_loss = self.eval_neg_probe.criterion(neg_logits, target, reduction='none')

            metrics['sparsity'] += mask.mean().item()
            metrics['pos_loss'] += pos_loss.mean().item()
            metrics['neg_loss'] += neg_loss.mean().item()
            metric_target = target.argmax(dim=1) if self.task_type == 'classification' and self.target_type == 'soft' else target
            metrics['pos_metric'] += self.eval_pos_probe.metric(pos_logits, metric_target).item()
            metrics['neg_metric'] += self.eval_neg_probe.metric(neg_logits, metric_target).item()

            if self.task_type == 'classification':
                metrics['fidplus_prob'] += fid_plus_prob(neg_logits, baseline_logits)
                metrics['fidminus_prob'] += fid_minus_prob(pos_logits, baseline_logits)
                metrics['fidplus_acc'] += fid_plus_acc(neg_logits, baseline_logits)
                metrics['fidminus_acc'] += fid_minus_acc(pos_logits, baseline_logits)
                metrics['fidplus'] += fid_plus_cat_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_cat_tv(pos_logits, baseline_logits)
            else:
                metrics['fidplus'] += fid_plus_reg_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_reg_tv(pos_logits, baseline_logits)

            if hasattr(data, 'true'):
                metrics['precision'] += precision_score(mask, data.true)
                metrics['recall'] += recall_score(mask, data.true)
                metrics['f1'] += f1_score(mask, data.true)
                metrics['pr_auc'] += pr_auc_score(probs, data.true)
                metrics['roc_auc'] += roc_auc_score(probs, data.true)

        for metric_name in metrics:
            metrics[metric_name] /= len(loader)
        return metrics

    def fit(self, train_loader, val_loader, epochs):
        if self.eval_pos_probe is None or self.eval_neg_probe is None:
            self.initialize_probes()
        best_pos_val_loss = float('inf')
        best_neg_val_loss = float('inf')
        pos_path, neg_path = self._checkpoint_paths()
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate_epoch(val_loader)
            if val_metrics['pos_loss'] < best_pos_val_loss:
                best_pos_val_loss = val_metrics['pos_loss']
                torch.save(self.eval_pos_probe.state_dict(), pos_path)
            if val_metrics['neg_loss'] < best_neg_val_loss:
                best_neg_val_loss = val_metrics['neg_loss']
                torch.save(self.eval_neg_probe.state_dict(), neg_path)
            yield epoch, train_metrics, val_metrics

    def load_best_probes(self):
        if self.eval_pos_probe is None or self.eval_neg_probe is None:
            raise RuntimeError('Call initialize_probes() before load_best_probes().')
        pos_path, neg_path = self._checkpoint_paths()
        self.eval_pos_probe.load_state_dict(torch.load(pos_path, weights_only=False))
        self.eval_neg_probe.load_state_dict(torch.load(neg_path, weights_only=False))
