"""Task losses and the original implementation's regularizer scaling."""

import torch
import torch.nn.functional as F
from torchsurv.loss.cox import neg_partial_log_likelihood


def classification_loss(expert_logits, target, assignments):
    """Apply routing to per-subject BCE before taking the batch mean."""
    per_subject = F.binary_cross_entropy_with_logits(
        expert_logits, target[:, None].expand_as(expert_logits), reduction="none"
    )
    return (per_subject * assignments).sum(dim=1).mean()


def survival_loss(expert_log_risks, event, time, assignments):
    """Hard expert-specific Cox loss with straight-through risk gradients.

    assignments is the floating-point, hard Gumbel-softmax result. Mixing risks
    BEFORE indexing preserves its gradient path to the router. Forward values
    and risk-set membership remain identical to hard expert selection. The
    backward pass is a straight-through surrogate through selected risk values;
    discrete risk-set membership itself is held fixed during differentiation.

    Retains TorchSurv's Efron ties handling and per-expert mean reduction, then
    sums expert losses as in the source implementation (not paper Eq. 2 scaling).
    """
    routed_risk = (expert_log_risks * assignments).sum(dim=1)
    loss = routed_risk.sum() * 0.0
    for expert in range(assignments.shape[1]):
        members = assignments[:, expert].detach().bool()
        if members.sum() > 1 and event[members].any():
            loss = loss + neg_partial_log_likelihood(
                routed_risk[members], event[members], time[members],
                ties_method="efron", reduction="mean",
            )
    return loss


def router_regularization(logits, pseudo_labels):
    """Return balance, sparsity, guidance; exclude reference rows labeled -1."""
    selected = pseudo_labels >= 0
    if not selected.any():
        zero = logits.sum() * 0.0
        return zero, zero, zero
    logits = logits[selected]
    probabilities = logits.softmax(dim=1)
    average = probabilities.mean(dim=0)
    # Deliberately retain the extra 1/K factor from the original code.
    balance = (average * (average + 1e-8).log()).mean()
    sparsity = -(probabilities * (probabilities + 1e-8).log()).mean()
    guidance = F.cross_entropy(logits, pseudo_labels[selected])
    return balance, sparsity, guidance


def guidance_factor(epoch, decay_epochs=20, power=1.0):
    """Zero-based epoch: linear default 1, .95, ..., .05, 0 at epoch 20.

    power > 1 decays faster; 0 < power < 1 decays slower. A zero duration
    disables guidance, independently of the total training duration.
    """
    if decay_epochs == 0:
        return 0.0
    return max(0.0, 1.0 - epoch / decay_epochs) ** power
