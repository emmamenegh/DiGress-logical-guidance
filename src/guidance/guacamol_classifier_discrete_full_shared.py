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
def log_train_epoch_summary(epoch, logs, duration, task_names):
    print(f"Epoch {epoch} Summary (Train) — Duration: {duration:.1f}s")
    for task in task_names:
        print(f"  [{task}] "
              f"Acc: {logs.get(f'train/accuracy_{task}', float('nan')):.3f}, "
              f"F1: {logs.get(f'train/f1_{task}', float('nan')):.3f}, "
              f"AUROC: {logs.get(f'train/auroc_{task}', float('nan')):.3f}, "
              f"AUCPR: {logs.get(f'train/average_precision_{task}', float('nan')):.3f}, "
              f"BCE: {logs.get(f'train/bce_{task}', float('nan')):.4f}")
    print()

@rank_zero_only
def log_val_epoch_summary(epoch, logs, best_val_bce, is_new_best, task_names):
    print(f"Epoch {epoch} Summary (Validation)")

    for task in task_names:
        print(
            f"  [{task}] "
            f"Acc: {logs.get(f'val/accuracy_{task}', float('nan')):.3f}, "
            f"F1: {logs.get(f'val/f1_{task}', float('nan')):.3f}, "
            f"AUROC: {logs.get(f'val/auroc_{task}', float('nan')):.3f}, "
            f"AUCPR: {logs.get(f'val/average_precision_{task}', float('nan')):.3f}, "
            f"BCE: {logs.get(f'val/bce_{task}', float('nan')):.4f}"
        )

    # Compute mean BCE across tasks
    mean_bce = sum(logs[f"val/bce_{task}"] for task in task_names) / len(task_names)
    flag = "  --> New best mean BCE!" if is_new_best else ""
    print(f"\n  [MEAN] BCE: {mean_bce:.4f} | Best: {best_val_bce:.4f}{flag}\n")


class TrainBinaryMetrics(nn.Module):
    """
    Tracks and computes binary classification metrics (accuracy, AUROC, F1 score, average precision, 
    BCE loss, and precision-recall curves) independently for each task in a multi-head binary classification setup.
    """
    def __init__(self, task_names, prefix="train"):
        super().__init__()
        self.task_names = task_names
        self.prefix = prefix  # e.g., "train", "val", or "test"

        self.accuracy = nn.ModuleDict({task: BinaryAccuracy() for task in task_names})
        self.auroc = nn.ModuleDict({task: BinaryAUROC() for task in task_names})
        self.f1 = nn.ModuleDict({task: BinaryF1Score() for task in task_names})
        self.ap = nn.ModuleDict({task: BinaryAveragePrecision() for task in task_names})
        self.prec_recall_curve = nn.ModuleDict({task: BinaryPrecisionRecallCurve() for task in task_names})
        self.bce = nn.ModuleDict({task: MeanMetric() for task in task_names})

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates all metric trackers with a new batch of predictions and targets.

        Args:
            preds: Tensor of shape [B, T] — raw logits from the model.
            targets: Tensor of shape [B, T] — binary ground truth labels.
        """
        for i, task in enumerate(self.task_names):
            pred = preds[:, i]
            target = targets[:, i].float()

            self.accuracy[task].update(pred, target.int())
            self.auroc[task].update(pred, target.int())
            self.f1[task].update(pred, target.int())
            self.ap[task].update(pred, target.int())
            self.prec_recall_curve[task].update(pred, target.int())

            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
            self.bce[task].update(bce, weight=pred.numel())

    def compute(self):
        """
        Computes and returns all scalar metrics as a dictionary.
        """
        metrics = {}
        for task in self.task_names:
            metrics[f"{self.prefix}/accuracy_{task}"] = self.accuracy[task].compute()
            metrics[f"{self.prefix}/auroc_{task}"] = self.auroc[task].compute()
            metrics[f"{self.prefix}/f1_{task}"] = self.f1[task].compute()
            metrics[f"{self.prefix}/average_precision_{task}"] = self.ap[task].compute()
            metrics[f"{self.prefix}/bce_{task}"] = self.bce[task].compute()
            metrics[f"{self.prefix}/pr_curve_{task}"] = self.prec_recall_curve[task].compute()
        return metrics

    def log(self):
        """
        Logs all scalar metrics and precision-recall curves to Weights & Biases (wandb).
        """
        metrics = self.compute()
        log_dict = {}

        for key, value in metrics.items():
            if "pr_curve_" in key:
                precision, recall, thresholds = value 
                stage, _, task_key = key.partition("/")
                _, _, task_name = task_key.partition("_")

                log_dict[f"{stage}/pr_curve_{task_name}"] = wandb.plot.line_series(
                    xs = recall.cpu().tolist(),
                    ys = [precision.cpu().tolist()],
                    title= f"{stage.capitalize()} PR-curve – {task_name}",
                    xname= "Recall"
                )
            else:
                log_dict[key] = value

        rank_zero_only(wandb.log)(log_dict)

        return log_dict

    def reset(self):
        """
        Resets all metric trackers.
        """
        for task in self.task_names:
            self.accuracy[task].reset()
            self.auroc[task].reset()
            self.f1[task].reset()
            self.ap[task].reset()
            self.prec_recall_curve[task].reset()
            self.bce[task].reset()



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

        # buld a multi-head Transformer
        if self.cfg.general.guidance_target == "all":
            self.task_names = ["logP", "molW", "HBD", "HBA"]
        else:
            raise ValueError(f"Unsupported guidance_target: {self.cfg.general.guidance_target}")

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

        self.bce_loss_fn = F.binary_cross_entropy_with_logits  # use functional version
        self.best_val_bce = float("inf") 

        # binary classification metrics for multi-task
        self.metrics = nn.ModuleDict({
            "train_metrics": TrainBinaryMetrics(self.task_names, prefix="train"),
            "val_metrics"  : TrainBinaryMetrics(self.task_names, prefix="val"),
            "test_metrics" : TrainBinaryMetrics(self.task_names, prefix="test")
        })


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
            Computes BCEWithLogitsLoss per task for multi-head binary classification.
            Optionally logs per-task BCE to Weights & Biases.
            """
            total_loss = 0
            per_task_bce = {}

            for i, task in enumerate(self.task_names):
                task_loss = self.bce_loss_fn(pred.y[:, i], target[:, i].float())
                per_task_bce[task] = task_loss
                total_loss += task_loss

            mean_loss = total_loss / len(self.task_names)

            if log and pl.utilities.rank_zero._get_rank() == 0:
                log_dict = {
                    f"train_loss/bce_{task}": per_task_bce[task].item()
                    for task in self.task_names
                }
                log_dict["train_loss/bce_mean"] = mean_loss.item()
                wandb.log(log_dict, commit=True)

            return mean_loss

    def training_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.float()
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
        self.metrics["train_metrics"].update(pred.y, target)

        return {"loss": bce}

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        safe_print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.metrics["train_metrics"].reset()

    def on_train_epoch_end(self) -> None:
        logs = self.metrics["train_metrics"].log()
        log_train_epoch_summary(self.current_epoch, logs, time.time() - self.start_epoch_time, self.task_names)

        self.metrics["train_metrics"].reset()

    def on_validation_epoch_start(self) -> None:
        self.metrics["val_metrics"].reset()

    def validation_step(self, data, i):

        # input zero y to generate noised graphs
        target = data.y.float()
        data_y_zeroed = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data_y_zeroed, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # compute BCE loss
        bce = self.compute_loss(pred, target)
        # update validation metrics
        self.metrics["val_metrics"].update(pred.y, target)

        return {"val_loss": bce}

    def validation_epoch_end(self, outs) -> None:
        # Log and retrieve val metrics
        logs = self.metrics["val_metrics"].log() 

        # Compute mean BCE across tasks
        bce_keys = [k for k in logs if k.startswith("val/bce_")]
        val_bce = (sum(logs[k] for k in bce_keys) / len(bce_keys)).item()

        # Log mean BCE to Lightning
        self.log("val/bce", val_bce, prog_bar=True, sync_dist=True)
        # Update best BCE
        is_new_bce_best = val_bce < self.best_val_bce
        if is_new_bce_best:
            self.best_val_bce = val_bce
        # Print epoch summary
        log_val_epoch_summary(self.current_epoch, logs, self.best_val_bce, is_new_bce_best, self.task_names)

        # Reset val metrics
        self.metrics["val_metrics"].reset()

    def on_test_epoch_start(self) -> None:
        self.metrics["test_metrics"].reset()

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
