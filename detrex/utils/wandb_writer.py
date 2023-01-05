import wandb
import os
from detectron2.utils.events import EventWriter, get_event_storage


class WandBWriter(EventWriter):
    """
    Write all scalars to a WandB.
    """

    def __init__(self, cfg, window_size=20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.cfg = cfg.train.wandb
        wandb.init(
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name if cfg.wandb.name != "" else os.path.basename(cfg.train.output_dir.strip('/')),
            project=cfg.train.wandb.project,
            config=cfg
        )
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                wandb.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write