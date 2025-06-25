import gc
import time

import detectron2.checkpoint as checkpointer
import torch
from detectron2.config import instantiate
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm

from deepdisc.astrodet import detectron as detectron_addons

class LazyAstroTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)

        # Borrowed from DefaultTrainer constructor
        # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        self.checkpointer = checkpointer.DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
        )
        # load weights
        self.checkpointer.load(cfg.train.init_checkpoint)

        # record loss over iteration
        self.lossList = []
        self.lossdict_epochs = {}
        self.vallossList = []
        self.vallossdict_epochs = {}

        self.period = 20
        self.iterCount = 0

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # self.scheduler = instantiate(cfg.lr_multiplier)
        self.valloss = 0
        self.vallossdict={}

    # Note: print out loss over p iterations
    def set_period(self, p):
        self.period = p

    # Copied directly from SimpleTrainer, add in custom manipulation with the loss
    # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html#SimpleTrainer
    def run_step(self):
        self.iterCount = self.iterCount + 1
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        # Note: in training mode, model() returns loss
        start = time.perf_counter()
        loss_dict = self.model(data)
        loss_time = time.perf_counter() - start

    
        ld = {
             k: v.detach().cpu().item() if (isinstance(v, torch.Tensor) and v.numel()==1)  else v.tolist()
             for k, v in loss_dict.items()
        }

        self.lossdict_epochs[str(self.iter+1)] = ld

        # print('Loss dict',loss_dict)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
            all_losses = [l.cpu().detach().item() for l in loss_dict.values()]
        self.optimizer.zero_grad()
        losses.backward()

        # self._write_metrics(loss_dict,data_time)

        self.optimizer.step()

        self.lossList.append(losses.cpu().detach().numpy())
        if self.iter % self.period == 0 and comm.is_main_process():
            # print("Iteration: ", self.iterCount, " time: ", data_time," loss: ",losses.cpu().detach().numpy(), "val loss: ",self.valloss, "lr: ", self.scheduler.get_lr())
            print(
                "Iteration: ",
                self.iter,
                " data time: ",
                data_time,
                " loss time: ",
                loss_time,
                loss_dict.keys(),
                all_losses,
                "val loss: ",
                self.valloss,
                "lr: ",
                self.scheduler.get_lr(),
            )

        #del data
        #gc.collect()
        #torch.cuda.empty_cache()

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    def add_val_loss(self, val_loss):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossList.append(val_loss)

    def add_val_loss_dict(self, val_loss_dict):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossdict_epochs[str(self.iter+1)] = val_loss_dict


class LazyAstroEvaluator(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)

        # Borrowed from DefaultTrainer constructor
        # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        self.checkpointer = checkpointer.DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
        )
        # load weights
        self.checkpointer.load(cfg.train.init_checkpoint)

        # record loss over iteration
        self.lossList = []
        self.lossdict_epochs = {}
        self.vallossList = []
        self.vallossdict_epochs = {}

        self.period = 20
        self.iterCount = 0

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # self.scheduler = instantiate(cfg.lr_multiplier)
        self.valloss = 0
        self.vallossdict={}

    # Note: print out loss over p iterations
    def set_period(self, p):
        self.period = p

    # Copied directly from SimpleTrainer, add in custom manipulation with the loss
    # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html#SimpleTrainer
    def run_step(self):
        # Copying inference_on_dataset from evaluator.py
        #total = len(self.data_loader)
        #num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        losses_dicts =[]
        with torch.no_grad():
            for idx, inputs in enumerate(self.data_loader):
                '''
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                '''
                metrics_dict = self.model(inputs)
                losses_dicts.append(metrics_dict)
                
            #losses.append(loss_batch)
            #losses_dicts.append(metrics_dict)
        #mean_loss = np.mean(losses)
        self.losses_dicts = losses_dicts
        
        
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    def add_val_loss(self, val_loss):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossList.append(val_loss)

    def add_val_loss_dict(self, val_loss_dict):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossdict_epochs[str(self.iterCount)] = val_loss_dict


def return_lazy_trainer(model, loader, optimizer, cfg, hooklist):
    """Return a trainer for models built on LazyConfigs

    Parameters
    ----------
    model : torch model
        pointer to file
    loader : detectron2 data loader

    optimizer : detectron2 optimizer

    cfg : .py file
        The LazyConfig used to build the model, and also stores config vals for data loaders

    hooklist : list
        The list of hooks to use for the trainer

    Returns
    -------
        trainer
    """
    trainer = LazyAstroTrainer(model, loader, optimizer, cfg)
    trainer.register_hooks(hooklist)
    return trainer


def return_savehook(output_name, save_period):
    """Returns a hook for saving the model

    Parameters
    ----------
    output_name : str
        name of output file to save

    Returns
    -------
        a SaveHook
    """
    saveHook = detectron_addons.SaveHook(save_period)
    saveHook.set_output_name(output_name)
    return saveHook


def return_schedulerhook(optimizer):
    """Returns a hook for the learning rate

    Parameters
    ----------
    optimizer : detectron2 optimizer
        the optimizer that controls the learning rate

    Returns
    -------
        a CustomLRScheduler hook
    """
    schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
    return schedulerHook


def return_evallosshook(val_per, model, test_loader):
    """Returns a hook for evaulating the loss

    Parameters
    ----------
    val_per : int
        the frequency with which to calculate validation loss
    model: torch.nn.module
        the model
    test_loader: data loader
        the loader to read in the eval data

    Returns
    -------
        a LossEvalHook
    """
    lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
    return lossHook


def return_optimizer(cfg):
    """Returns an optimizer for training

    Parameters
    ----------
    cfg : .py file
        The LazyConfig used to build the model

    Returns
    -------
        a pytorch optimizer
    """
    optimizer = instantiate(cfg.optimizer)
    return optimizer
