import os, time, datetime, shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence, Callable, Any, Dict

import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from logging import Logger
from .logger import create_logger
from .misc import save_args


def get_parser(description: str = None):
    """
    get ArgumentParser with necessary arguments for trainer

    Args:
        description: for ArgumentParser
    """
    parser = ArgumentParser(description=description)
    
    parser.add_argument("--output", default=None, type=str, help="path of output directory")
    parser.add_argument("--no_tensorboard", action="store_true", default=False, help="use if tensorboard is not available or not needed")

    parser.add_argument("--batch_size", default=1, type=int, help="batch size **per GPU** for training")
    parser.add_argument("--max_epochs", default=1, type=int, help="max epochs to train")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency (epoch)")
    parser.add_argument("--scheduler_type", default="epoch", choices=["epoch", "iteration"], type=str, help="type of lr scheduler")
    
    parser.add_argument("--distributed", action="store_true", default=False, help="start distributed training")
    parser.add_argument("--nnodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--master_addr", default="localhost", type=str, help="master address for init distributed")
    parser.add_argument("--master_port", default="62674", type=str, help="master address for init distributed")

    parser.add_argument("--amp", action="store_true", default=False, help="use automatic mixed precision (amp) for training")
    parser.add_argument("--workers", default=4, type=int, help="number of workers for dataloader")

    return parser


def main_runner(main_worker: Callable[[int, Namespace], Any], args: Namespace):
    """
    Args:
        main_worker: to start training, **must** use `(local_rank, args)` as input
        args: **must** with the necessary arguments in `get_parser` 
    """
    assert torch.cuda.is_available()

    if args.output is not None:
        args.output = Path(args.output)
        args.output.mkdir(parents=True, exist_ok=True)

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        
        assert args.nnodes >= 1, "`nnodes` should be 1 at least."
        print(f"Found {args.ngpus_per_node} GPUs this node.")
        if args.nnodes > 1:
            print(f"Please make sure the number of GPUs is the same as other nodes!")

        args.world_size = args.ngpus_per_node * args.nnodes
        save_args(args)
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        save_args(args)
        main_worker(0, args)


def init_main_worker(local_rank: int, args: Namespace):
    """
    necessary initialization for main_worker
    """
    if args.distributed:
        args.rank = args.node_rank * args.ngpus_per_node + local_rank
        dist.init_process_group(backend="nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}", world_size=args.world_size, rank=args.rank)
        args.local_rank = local_rank
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = args.local_rank = local_rank
    
    logger = create_logger(name=f"Rank{args.rank}", output_path=(args.output / f"log_rank{args.rank}.log") if args.output is not None else None)
    args.logger = logger

    logger.info(f"Node {args.node_rank}, Local Rank {args.local_rank}: {torch.cuda.get_device_name(args.local_rank)}")


def get_loader(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        args: Namespace
):
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if args.distributed else None

    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=valid_sampler,
            pin_memory=True
        )
    
    if args.distributed:
        args.train_valid_length = (len(train_dataset) - args.rank - 1) // train_sampler.num_replicas + 1
        args.valid_valid_length = (len(valid_dataset) - args.rank - 1) // valid_sampler.num_replicas + 1
    
    return train_loader, valid_loader


def run_training(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: optim.Optimizer,
        prepare_train_batch_func: Callable[[Any, Namespace], Dict[str, Any]],
        prepare_valid_batch_func: Callable[[Any, Namespace], Dict[str, Any]],
        loss_func: nn.modules.loss._Loss,
        metric_func: Callable[[Tensor, Tensor], Tensor],
        args: Namespace,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        scaler: GradScaler = None,
        start_epoch: int = 0,
        global_step: int = 0
):
    """
    Args:
        prepare_train_batch_func: Prepare batch for training. **Must** use `(batch, args)` as input and output as `{"inputs": ..., "targets": ...}`
        prepare_valid_batch_func: Prepare batch for validation. **Must** use `(batch, args)` as input and output as `{"inputs": ..., "targets": ...}`
        loss_func: Compute loss in training. Will use as `loss_func(model(prepared_batch["inputs"]), prepared_batch["targets"])`
        metric_func: Compute metric in validation. Will use as `metric_func(model(prepared_batch["inputs"]), prepared_batch["targets"])`
        scaler: If args.amp is True and scaler is None, will use `torch.cuda.amp.GradScaler()` as default
    """
    logger: Logger = args.logger

    args.no_tensorboard = args.no_tensorboard or args.output is None
    if args.rank == 0 and not args.no_tensorboard:
        try:
            writer = SummaryWriter(log_dir=args.output / "tensorboard")
            args.writer = writer
            logger.info(f"Writing tensorboard to {writer.get_logdir()}")
        except Exception as e:
            logger.exception(e)
            args.no_tensorboard = True
            logger.warning("Tensorboard is not available. `no_tensorboard` has been set.")

    if args.amp and scaler is None:
        scaler = GradScaler()
    
    if args.rank == 0:
        valid_metric_max = 0.0
        training_start_time = phase_start_time = time.perf_counter()

    for epoch in range(start_epoch, args.max_epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            prepare_batch_func=prepare_train_batch_func,
            loss_func=loss_func,
            args=args,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step
        )

        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.max_epochs:
            if args.distributed:
                dist.barrier()
            
            valid_metric = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                prepare_batch_func=prepare_valid_batch_func,
                metric_func=metric_func,
                args=args
            )
            
            if args.rank == 0:
                if not args.no_tensorboard:
                    writer.add_scalar("Validation/Metric", valid_metric, epoch + 1)
                
                is_new_best = valid_metric > valid_metric_max
                if is_new_best:
                    valid_metric_max = valid_metric
                    logger.info(f"Validation ({epoch + 1} / {args.max_epochs} Epochs) new best metric: {valid_metric:2.6f}")
                else:
                    logger.info(f"Validation ({epoch + 1} /{args.max_epochs} Epochs) metric: {valid_metric:2.6f}, best: {valid_metric_max:2.6f}")
                
                save_checkpoint(
                    save_path=args.output / "model_final.pt",
                    model=model,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                    metric=valid_metric,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler
                )

                if is_new_best:
                    shutil.copyfile(args.output / "model_final.pt", args.output / "model_best.pt")
                    logger.debug("New best model has been copied to model_best.pt")
                
                phase_end_time = time.perf_counter()
                logger.info(
                    f"Total time used: {datetime.timedelta(seconds=int(phase_end_time - training_start_time))}; Expected time left: {datetime.timedelta(seconds=int((phase_end_time - phase_start_time) / args.val_every * (args.max_epochs - epoch - 1)))}"
                )
                phase_start_time = phase_end_time


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        prepare_batch_func: Callable[[Any, Namespace], Dict[str, Any]],
        loss_func: nn.modules.loss._Loss,
        args: Namespace,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        scaler: GradScaler = None,
        epoch: int = 0,
        global_step: int = 0,
):
    model.train()
    epoch_loss = 0
    step = 0
    
    if args.rank == 0:
        epoch_iterator = tqdm(train_loader, desc=f"Training ({epoch + 1:d} / {args.max_epochs:d} Epochs) (loss={0:2.5f})", dynamic_ncols=True)
    else:
        epoch_iterator = train_loader

    if args.rank == 0 and not args.no_tensorboard:
        writer: SummaryWriter = args.writer
        if scheduler is not None and args.scheduler_type == "epoch":
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
    
    for batch in epoch_iterator:
        if scheduler is not None and args.scheduler_type == "iteration":
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)

        optimizer.zero_grad(set_to_none=True)
        
        prepared_batch = prepare_batch_func(batch, args)
        
        with autocast(enabled=args.amp):
            batch_preds = model(prepared_batch["inputs"])
            loss: Tensor = loss_func(batch_preds, prepared_batch["targets"])
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss.unsqueeze(0)])[0]
            step_loss = sum(loss_list) / len(loss_list)
        else:
            step_loss = loss.item()
        epoch_loss += step_loss

        if args.rank == 0 and not args.no_tensorboard:
            writer.add_scalar("Train/Loss(step)", step_loss, global_step + step)

        step += 1
        if args.rank == 0:
            epoch_iterator.set_description(f"Training ({epoch + 1:d} / {args.max_epochs:d} Epochs) (loss={epoch_loss / step:2.5f})")
        
        if scheduler is not None and args.scheduler_type == "iteration":
            scheduler.step()
    
    if scheduler is not None and args.scheduler_type == "epoch":
        scheduler.step()
    
    if args.rank == 0 and not args.no_tensorboard:
        writer.add_scalar("Train/Loss(epoch)", epoch_loss / step, global_step + step - 1)
    
    return global_step + step


@torch.no_grad()
def valid_epoch(
    model: nn.Module,
    valid_loader: DataLoader,
    prepare_batch_func: Callable[[Any, Namespace], Dict[str, Any]],
    metric_func: Callable[[Tensor, Tensor], Tensor],
    args: Namespace
):
    model.eval()
    
    metric_sum = 0.0
    valid_num = 0
    
    if args.rank == 0:
        epoch_iterator = tqdm(valid_loader, desc=f"Validate (metric={0:2.5f})", dynamic_ncols=True)
    else:
        epoch_iterator = valid_loader

    for idx, batch in enumerate(epoch_iterator):
        prepared_batch = prepare_batch_func(batch, args)

        with autocast(enabled=args.amp):
            batch_preds = model(prepared_batch["inputs"])

        metrics = metric_func(batch_preds, prepared_batch["targets"])

        if args.distributed:
            is_valid_list = [idx * args.batch_size + j < args.valid_valid_length for j in range(metrics.shape[0])]
            metrics_list = distributed_all_gather([metrics], is_valid=is_valid_list)[0]
            metric_sum += sum(metrics_list)
            valid_num += len(metrics_list)
        else:
            valid_num += metrics.shape[0]
            metric_sum += metrics.sum().item()
        
        if args.rank == 0:
            epoch_iterator.set_description(f"Validate (metric={metric_sum / valid_num:2.6f})")

    return metric_sum / valid_num


@torch.no_grad()
def distributed_all_gather(
    tensor_list: Sequence[Tensor],
    is_valid: Sequence[bool] = None,
    world_size: int = None,
    no_barrier: bool = False
):
    """
    Args:
        tensor_list: [Tensor(Batch), x N]
        is_valid: [bool(Batch)], is_valid[b] is for tensor_list[0:N][b]
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if is_valid is not None:
        is_valid = torch.as_tensor(is_valid, dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        dist.barrier()

    tensor_list_out = []

    if is_valid is not None:
        is_valid_cat = [torch.zeros_like(is_valid) for _ in range(world_size)]
        dist.all_gather(is_valid_cat, is_valid)
        is_valid_cat = torch.cat(is_valid_cat, dim=0)  # concat in dim Batch
    
    for tensor in tensor_list:

        gather_cat = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_cat, tensor)
        gather_cat = torch.cat(gather_cat, dim=0)  # concat in dim Batch

        if is_valid is None:
            is_valid_cat = torch.ones(gather_cat.shape[0], dtype=torch.bool)
        
        gather_list = [g.item() for g, v in zip(gather_cat, is_valid_cat) if v]
        
        tensor_list_out.append(gather_list)
    
    return tensor_list_out


def load_checkpoint(
        checkpoint_path: os.PathLike | str,
        model: nn.Module,
        logger: Logger,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        scaler: GradScaler = None,
        start_dict: dict = {}
):
    """
    Args:
        model: necessary
        optimizer: set to None if not needed
        scheduler: set to None if not needed
        scaler: set to None if not needed
        start_dict: support keys `epoch` and `global_step`
    """
    checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint.keys():
        scaler.load_state_dict(checkpoint["scaler"])

    if "epoch" in start_dict.keys() and "epoch" in checkpoint.keys():
        start_dict["epoch"] = checkpoint["epoch"]
    if "global_step" in start_dict.keys() and "global_step" in checkpoint.keys():
        start_dict["global_step"] = checkpoint["global_step"]
    
    logger.info(f"Resumed from {checkpoint_path}, metric: {checkpoint['metric']}")


def save_checkpoint(
        save_path: os.PathLike | str,
        model: nn.Module,
        epoch: int,
        global_step: int,
        args: Namespace,
        metric: float = 0.0,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        scaler: GradScaler = None
):
    save_dict = {"epoch": epoch, "global_step": global_step, "metric": metric}

    save_dict["model"] = model.module.state_dict() if args.distributed else model.state_dict()

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()

    torch.save(save_dict, save_path)
    
    logger: Logger = args.logger
    logger.debug(f"Saving checkpoint {save_path}")
