"""Checkpoint persistence and periodic training resume state."""

import glob
import json
import logging
import os
import pickle
import re

log = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints and the latest periodic resume state."""

    state_name = "train_state.json"
    epoch_pattern = re.compile(
        r"(?:epoch_(\d+)|best_ep(\d+)_val[0-9.]+|best_mini_ep(\d+)_score[0-9.]+)\.pkl"
    )

    def __init__(self, checkpoint_dir):
        self.dir = checkpoint_dir

    def periodic_path(self, epoch):
        return os.path.join(self.dir, f"epoch_{epoch:03d}.pkl")

    def best_path(self, epoch, val_loss):
        return os.path.join(self.dir, f"best_ep{epoch:03d}_val{val_loss:.6f}.pkl")

    def best_mini_path(self, epoch, mini_score):
        return os.path.join(
            self.dir,
            f"best_mini_ep{epoch:03d}_score{mini_score:.4f}.pkl",
        )

    def state_path(self):
        return os.path.join(self.dir, self.state_name)

    def model_paths(self, pattern):
        paths = glob.glob(os.path.join(self.dir, pattern))
        return sorted(
            path
            for path in paths
            if not path.endswith("_optim.pkl") and not path.endswith("_ema.pkl")
        )

    @staticmethod
    def optimizer_path(model_path):
        root, extension = os.path.splitext(model_path)
        return f"{root}_optim{extension}"

    @staticmethod
    def ema_path(model_path):
        root, extension = os.path.splitext(model_path)
        return f"{root}_ema{extension}"

    @classmethod
    def epoch_from_path(cls, path):
        match = cls.epoch_pattern.fullmatch(os.path.basename(path))
        if not match:
            raise ValueError(f"Cannot infer epoch from checkpoint name: {path}")
        return int(next(group for group in match.groups() if group is not None))

    @staticmethod
    def write_json(path, data):
        """Atomically write indented JSON followed by one newline."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w") as file:
            json.dump(data, file, indent=2, sort_keys=False)
            file.write("\n")
        os.replace(temporary_path, path)

    @staticmethod
    def remove(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def state(
        checkpoint_path,
        epoch,
        best_loss,
        best_epoch,
        best_mini_score,
        best_mini_epoch,
        history,
    ):
        """Build the human-readable resume-state mapping."""
        state = {
            "checkpoint": os.path.basename(checkpoint_path),
            "epoch": epoch,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_mini_score": best_mini_score if best_mini_epoch else None,
            "best_mini_epoch": best_mini_epoch,
            "history": history,
        }
        return state

    @staticmethod
    def read_state(state):
        """Return resume trackers from a persisted state mapping."""
        return (
            state["checkpoint"],
            state["epoch"],
            state["best_loss"],
            state.get("best_epoch", 0),
            state["best_mini_score"],
            state["best_mini_epoch"],
            state.get("history", []),
        )

    def save_optimizer(self, optimizer, model_path):
        if not hasattr(optimizer, "state_dict"):
            log.warning(
                "  Optimizer does not expose state_dict(); resume is weights-only"
            )
            return
        try:
            with open(self.optimizer_path(model_path), "wb") as file:
                pickle.dump(optimizer.state_dict(), file)
        except Exception as error:
            log.warning(
                "  Failed to save optimizer state (%s); resume is weights-only",
                error,
            )

    def load_optimizer(self, optimizer, model_path):
        path = self.optimizer_path(model_path)
        if not os.path.exists(path):
            log.warning("  Optimizer state missing - resume is weights-only")
            return
        if not hasattr(optimizer, "load_state_dict"):
            log.warning(
                "  Optimizer does not expose load_state_dict(); resume is weights-only"
            )
            return
        try:
            with open(path, "rb") as file:
                optimizer.load_state_dict(pickle.load(file))
            log.info("  Loaded optimizer state %s", path)
        except Exception as error:
            log.warning(
                "  Failed to load optimizer state (%s); resume is weights-only",
                error,
            )

    def save_model(self, model, path):
        os.makedirs(self.dir, exist_ok=True)
        model.save(path)
        return path

    def save_resume_state(
        self,
        checkpoint_path,
        epoch,
        best_loss,
        best_epoch,
        best_mini_score,
        best_mini_epoch,
        history,
    ):
        self.write_json(
            self.state_path(),
            self.state(
                checkpoint_path,
                epoch,
                best_loss,
                best_epoch,
                best_mini_score,
                best_mini_epoch,
                history,
            ),
        )

    def save_periodic(
        self,
        model,
        optimizer,
        ema,
        epoch,
        best_loss,
        best_epoch,
        best_mini_score,
        best_mini_epoch,
        history,
    ):
        path = self.save_model(model, self.periodic_path(epoch))
        self.save_optimizer(optimizer, path)
        if ema is not None:
            ema.save(self.ema_path(path))
        self.save_resume_state(
            path,
            epoch,
            best_loss,
            best_epoch,
            best_mini_score,
            best_mini_epoch,
            history,
        )
        log.info("  Periodic checkpoint saved -> %s", path)
        return path

    def save_best(self, model, epoch, val_loss):
        path = self.save_model(model, self.best_path(epoch, val_loss))
        for old_path in self.model_paths("best_ep*.pkl"):
            if old_path != path:
                self.remove(old_path)
                self.remove(self.optimizer_path(old_path))
        log.info("  Best val checkpoint saved -> %s", path)
        return path

    def save_best_mini(self, model, epoch, score):
        path = self.save_model(model, self.best_mini_path(epoch, score))
        for old_path in self.model_paths("best_mini_ep*.pkl"):
            if old_path != path:
                self.remove(old_path)
                self.remove(self.optimizer_path(old_path))
        log.info("  Best mini checkpoint saved -> %s", path)
        return path

    def load(self, model, optimizer, ema, path):
        """Restore a periodic checkpoint and return epoch and best trackers."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        epoch = self.epoch_from_path(path)
        model.load(path)
        self.load_optimizer(optimizer, path)
        if ema is not None:
            ema_path = self.ema_path(path)
            if os.path.exists(ema_path):
                ema.load(ema_path)
                log.info("  Loaded EMA state %s", ema_path)
            else:
                ema.reset()
                log.warning("  EMA state missing - initialized from loaded weights")

        state_path = os.path.join(os.path.dirname(path) or ".", self.state_name)
        if not os.path.exists(state_path):
            log.warning("  State file missing - best trackers reset")
            return epoch + 1, float("inf"), 0, -float("inf"), 0, []

        with open(state_path) as file:
            state = json.load(file)
        (
            checkpoint,
            state_epoch,
            best_loss,
            best_epoch,
            best_mini_score,
            best_mini_epoch,
            history,
        ) = self.read_state(state)

        if checkpoint != os.path.basename(path):
            log.warning("  State file is for %s - best trackers reset", checkpoint)
            return epoch + 1, float("inf"), 0, -float("inf"), 0, []
        if state_epoch != epoch:
            log.warning("  State epoch mismatch - best trackers reset")
            return epoch + 1, float("inf"), 0, -float("inf"), 0, []

        if best_mini_score is None:
            best_mini_score = -float("inf")
        log.info("  Loaded epoch %s, best_val_loss=%.6f", epoch, best_loss)
        return (
            epoch + 1,
            best_loss,
            best_epoch,
            best_mini_score,
            best_mini_epoch,
            history,
        )
