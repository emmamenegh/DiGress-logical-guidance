import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import time
import wandb
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryPrecisionRecallCurve
from torchmetrics import MeanMetric

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()

# for logging messages to the output files (not W&B logs)
@rank_zero_only
def safe_print(*args, **kwargs):
    print(*args, **kwargs)

@rank_zero_only
def log_train_epoch_summary(epoch, logs, duration):
    print(f"Epoch {epoch}: "
          f"Acc: {logs['train/accuracy']:.3f}, "
          f"F1: {logs['train/f1']:.3f}, "
          f"AUROC: {logs['train/auroc']:.3f}, "
          f"AUCPR: {logs['train/aucpr']:.3f}, "
          f"BCE: {logs['train/bce']:.4f}, "
          f"Time: {duration:.1f}s")
    
@rank_zero_only
def log_val_epoch_summary(epoch, logs, best_bce, is_new_best):
    print(f"Epoch {epoch}: "
          f"Val Acc: {logs['val/accuracy']:.3f}, "
          f"F1: {logs['val/f1']:.3f}, "
          f"AUROC: {logs['val/auroc']:.3f}, "
          f"AUCPR: {logs['val/aucpr']:.3f}, "
          f"BCE: {logs['val/bce']:.4f} \t Best BCE: {best_bce:.4f}")
    if is_new_best:
        print(f"New best val BCE: {best_bce:.4f}")


class TrainBinaryMetrics(nn.Module):
    """ Binary classification training/validation/testing metrics manager with BCE loss tracking. """
    def __init__(self):
        super().__init__()
        self.train_accuracy = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.train_f1 = BinaryF1Score()
        self.train_aucpr = BinaryAveragePrecision()
        self.train_pr_curve = BinaryPrecisionRecallCurve()
        self.train_bce = MeanMetric()

        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_aucpr = BinaryAveragePrecision()
        self.val_pr_curve = BinaryPrecisionRecallCurve()
        self.val_bce = MeanMetric()

        self.test_accuracy = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_aucpr = BinaryAveragePrecision()
        self.test_pr_curve = BinaryPrecisionRecallCurve()
        self.test_bce = MeanMetric()

    def forward(self, preds, targets, stage='train', bce_loss=None):
        """ Update metrics given preds and targets.
        Args:
            preds: logits (raw outputs)
            targets: true binary labels
            bce_loss: (optional) BCE loss value for tracking
            stage: 'train', 'val', or 'test'
        """
        preds = torch.sigmoid(preds)  # turn logits into probabilities

        if stage == 'train':
            self.train_accuracy.update(preds, targets.int())
            self.train_f1.update(preds, targets.int())
            self.train_auroc.update(preds, targets.int())
            self.train_aucpr.update(preds, targets.int())
            self.train_pr_curve.update(preds, targets.int())
            if bce_loss is not None:
                self.train_bce.update(bce_loss)
        elif stage == 'val':
            self.val_accuracy.update(preds, targets.int())
            self.val_f1.update(preds, targets.int())
            self.val_auroc.update(preds, targets.int())
            self.val_aucpr.update(preds, targets.int())
            self.val_pr_curve.update(preds, targets.int())
            if bce_loss is not None:
                self.val_bce.update(bce_loss)
        elif stage == 'test':
            self.test_accuracy.update(preds, targets.int())
            self.test_f1.update(preds, targets.int())
            self.test_auroc.update(preds, targets.int())
            self.test_aucpr.update(preds, targets.int())
            self.test_pr_curve.update(preds, targets.int())
            if bce_loss is not None:
                self.test_bce.update(bce_loss)

    def reset(self, stage='train'):
        """ Reset the metrics. """
        if stage == 'train':
            self.train_accuracy.reset()
            self.train_f1.reset()
            self.train_auroc.reset()
            self.train_aucpr.reset()
            self.train_pr_curve.reset()
            self.train_bce.reset()
        elif stage == 'val':
            self.val_accuracy.reset()
            self.val_f1.reset()
            self.val_auroc.reset()
            self.val_aucpr.reset()
            self.val_pr_curve.reset()
            self.val_bce.reset()
        elif stage == 'test':
            self.test_accuracy.reset()
            self.test_f1.reset()
            self.test_auroc.reset()
            self.test_aucpr.reset()
            self.test_pr_curve.reset()
            self.test_bce.reset()

    def compute(self, stage='train'):
        """ Compute the metrics. """
        if stage == 'train':
            return {
                'train_accuracy': self.train_accuracy.compute(),
                'train_f1': self.train_f1.compute(),
                'train_auroc': self.train_auroc.compute(),
                'train_aucpr': self.train_aucpr.compute(),
                'train_pr_curve': self.train_pr_curve.compute(),
                'train_bce': self.train_bce.compute()
            }
        elif stage == 'val':
            return {
                'val_accuracy': self.val_accuracy.compute(),
                'val_f1': self.val_f1.compute(),
                'val_auroc': self.val_auroc.compute(),
                'val_aucpr': self.val_aucpr.compute(),
                'val_pr_curve': self.val_pr_curve.compute(),
                'val_bce': self.val_bce.compute()
            }
        elif stage == 'test':
            return {
                'test_accuracy': self.test_accuracy.compute(),
                'test_f1': self.test_f1.compute(),
                'test_auroc': self.test_auroc.compute(),
                'test_aucpr': self.test_aucpr.compute(),
                'test_pr_curve': self.test_pr_curve.compute(),
                'test_bce': self.test_bce.compute()
            }

    def log(self, stage='train'):
        """ Compute and log the metrics automatically. """
        metrics = self.compute(stage=stage)

        log_dict = {}

        for key, value in metrics.items():
            if key.endswith("pr_curve"):
                # value is a tuple: (precision, recall, thresholds)
                precision, recall, _ = value
                stage = key.split("_", 1)[0]
                log_dict[f"{stage}/pr_curve"] = wandb.plot.line_series(
                    xs=recall.cpu().tolist(),
                    ys=[precision.cpu().tolist()],
                    title=f"{stage.capitalize()} Precision-Recall Curve",
                    xname="Recall"
                )
            else:
                # Correctly turn 'val_bce' into 'val/bce'
                parts = key.split('_', 1)
                if len(parts) == 2:
                    log_key = f"{parts[0]}/{parts[1]}"
                else:
                    log_key = key  # fallback
                log_dict[log_key] = value

        if pl.utilities.rank_zero._get_rank() == 0:
            wandb.log(log_dict)

        return log_dict

class GuacamolClassifierDiscrete(pl.LightningModule): # this class will be used for any classifier model
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y'] # binary output with BCEWithLogitsLoss
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # only used for diffusion model training
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric() # logp: log-probability of model outputs
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # the prediction head is already built in the Transformer
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        # Marginal transition model
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        safe_print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.best_val_bce = 1e8

        # binary classification metrics
        self.metrics = TrainBinaryMetrics()


    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=1e-12)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": None # ExponentialLR does not require a monitored metric
                },
            }
        
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def compute_loss(self, pred, target, log: bool = False):
        """
        Computes BCEWithLogitsLoss between prediction and target.
        Used for train, val.
        """
        bce = self.bce_loss_fn(pred.y, target)

        if log and pl.utilities.rank_zero._get_rank() == 0:
            wandb.log({"train_loss/batch_bce": bce.item()}, commit=True)
        
        return bce

    def training_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.unsqueeze(1).float()
        data_y_zeroed = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data_y_zeroed, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # compute BCE loss
        bce = self.compute_loss(pred, target, log=i % self.log_every_steps == 0)

        # update training metrics
        self.metrics(pred.y, target, stage="train", bce_loss=bce) # forward method

        return {"loss": bce}

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        safe_print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.metrics.reset(stage="train")

    def on_train_epoch_end(self) -> None:
        logs = self.metrics.log(stage="train")
        log_train_epoch_summary(self.current_epoch, logs, time.time() - self.start_epoch_time)

        self.metrics.reset(stage="train")

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset(stage="val")

    def validation_step(self, data, i):

        # input zero y to generate noised graphs
        target = data.y.unsqueeze(1).float()
        data_y_zeroed = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data_y_zeroed, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # compute BCE loss
        bce = self.compute_loss(pred, target)
        # update validation metrics
        self.metrics(pred.y, target, stage="val", bce_loss=bce) # forward method

        return {"val_loss": bce}

    def validation_epoch_end(self, outs) -> None:
        logs = self.metrics.log(stage="val")
        val_bce = logs["val/bce"]

        self.log("val/bce", val_bce, prog_bar=True, sync_dist=True)

        is_new_bce_best = val_bce < self.best_val_bce
        if is_new_bce_best:
            self.best_val_bce = val_bce
        log_val_epoch_summary(self.current_epoch, logs, self.best_val_bce, is_new_bce_best)

        self.metrics.reset(stage="val")

    def on_test_epoch_start(self) -> None:
        self.metrics.reset(stage="test")

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']

        assert extra_X.shape[-1] == 0, 'The classifier model should not be used with extra features'
        assert extra_E.shape[-1] == 0, 'The classifier model should not be used with extra features'
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)
