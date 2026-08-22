"""Gradient clipping and parameter EMA for training."""

import copy

import jittor as jt


class ModelEMA:
    """Maintain a frozen evaluation model with parameter-wise EMA updates."""

    def __init__(self, model, decay):
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.stop_grad()
        current = model.state_dict()
        average = self.model.state_dict()
        if current.keys() != average.keys():
            raise ValueError("EMA model does not match source model")
        self._state_pairs = tuple((current[name], average[name]) for name in current)
        jt.sync_all()

    def update(self):
        """Update the moving average after one optimizer step."""
        with jt.no_grad():
            for current, average in self._state_pairs:
                value = current.detach()
                average.update(self.decay * average + (1.0 - self.decay) * value)
        jt.sync_all(False)

    def reset(self):
        """Reset the moving average to the model's current parameters."""
        for current, average in self._state_pairs:
            average.update(current.detach())
        jt.sync_all()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        state = jt.load(path)
        average = self.model.state_dict()
        if state.keys() != average.keys():
            raise ValueError(f"EMA checkpoint does not match model: {path}")
        for name, param in average.items():
            value = state[name]
            if not isinstance(value, jt.Var):
                value = jt.array(value)
            if value.shape != param.shape:
                raise ValueError(
                    f"EMA shape mismatch for {name}: "
                    f"expected {param.shape}, got {value.shape}"
                )
            param.update(value.stop_grad())
        jt.sync_all()


def clip_grad_norm(optimizer, max_norm):
    """Clip the global trainable-gradient L2 norm and return it on-device."""
    squared_norms = [
        (gradient * gradient).sum()
        for group in optimizer.param_groups
        for param, gradient in zip(group["params"], group["grads"])
        if not param.is_stop_grad()
    ]
    total_norm = jt.sqrt(sum(squared_norms)) if squared_norms else jt.float32(0.0)
    coefficient = jt.minimum(float(max_norm) / (total_norm + 1e-6), 1.0)
    for group in optimizer.param_groups:
        for param, gradient in zip(group["params"], group["grads"]):
            if not param.is_stop_grad():
                gradient.update(gradient * coefficient)
    return total_norm.stop_grad()
