from torch.optim.lr_scheduler import ReduceLROnPlateau
from ultralytics.engine.trainer import BaseTrainer

class CustomLRScheduler:
    def __init__(self, monitor='val/mAP50', factor=0.5, patience=5, min_lr=1e-7):
        self.monitor = monitor
        self.scheduler = None
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = -float("inf")

    def on_train_start(self, trainer:BaseTrainer):
        # Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            trainer.optimizer,
            mode='max', # Monitor mAP (maximize)
            factor =self.factor,
            patience=self.patience,
            min_lr=self.min_lr
            ) 
        
    def on_fit_epoch_end(self, trainer:BaseTrainer):
        # Get metrics dictionary
        metrics = trainer.metrics
        # Extract The correct metrics(convert monitor name to key format)
        metric_key = self.monitor.replace('val/', 'metrics/')
        current_metric = metrics.get(metric_key, None)

        if current_metric is None:
            # Try alternative names
            alt_names = self.monitor.split('/')[-1]
            current_metric = metrics.get(alt_names, None)

        if current_metric is not None:
            # Update learning rate
            self.scheduler.step(current_metric)
            # Get Current LR
            current_lr = trainer.optimizer.param_groups[0]['lr']

            print(f"[LR Scheduler] Current metric: {current_metric:.4f} | LR: {current_lr:.7f}")

            if current_metric > self.best_metric:
                self.best_metric = current_metric
