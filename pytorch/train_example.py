import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from logging import Logger

from utils.trainer import get_parser, main_runner, init_main_worker, get_loader, run_training, load_checkpoint
from utils.misc import count_num_params

from model_toy import ToyNet
from dataset_toy import ToyDataset


def get_args():
    # get parser with necessary args for trainer
    parser = get_parser(description="Toy training")

    # add args you need below

    # for example
    parser.add_argument("--lr", default=1e-4, type=float, help="training learning rate")
    parser.add_argument('--resume', type=str, default=None, help='Training checkpoint path for resume')

    return parser.parse_args()


def main_worker(local_rank, args):
    
    # necessary initialization for main_worker
    init_main_worker(local_rank, args)
    # logger has been define in init_main_worker
    logger: Logger = args.logger

    if args.seed is None:
        # when determinism is not needed, enable it usually leads to faster runtime if input size is always the same
        torch.backends.cudnn.benchmark = True

    # define your dataset
    train_dataset = ToyDataset()
    valid_dataset = ToyDataset()

    # get dataloader
    train_loader, valid_loader = get_loader(train_dataset, valid_dataset, args)
    
    # define your model
    model = ToyNet().to(args.local_rank)
    
    # log number of parameters
    if args.rank == 0:
        count_num_params(model=model, logger=logger)
    
    # define needed objects
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scaler = GradScaler() if args.amp else None
    metric_func = toy_acc

    start_step = {"epoch": 0, "global_step": 0}

    # resume from checkpoint
    if args.resume is not None:
        load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            logger=logger,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            start_dict=start_step
        )

    if args.distributed:
        # convert when BatchNorm is used
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    run_training(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        prepare_train_batch_func=prepare_train_batch_func,
        prepare_valid_batch_func=prepare_valid_batch_func,
        loss_func=loss_func,
        metric_func=metric_func,
        args=args,
        scheduler=scheduler,
        scaler=scaler,
        start_epoch=start_step["epoch"],
        global_step=start_step["global_step"]
    )

    if args.distributed:
        dist.destroy_process_group()


def prepare_train_batch_func(batch, args):
    # based on ToyDataset
    images: Tensor = batch["image"]
    labels: Tensor = batch["label"]

    images = images.to(args.local_rank)
    labels = labels.to(args.local_rank, dtype=torch.long)

    return {"inputs": images, "targets": labels}


def prepare_valid_batch_func(batch, args):
    # based on ToyDataset
    images: Tensor = batch["image"]
    labels: Tensor = batch["label"]

    images = images.to(args.local_rank)
    labels = labels.to(args.local_rank, dtype=torch.long)

    return {"inputs": images, "targets": labels}


def toy_acc(inputs: Tensor, targets: Tensor):
    inputs = torch.argmax(inputs, dim=1)
    acc = (inputs == targets).float()
    return acc


if __name__ == "__main__":
    main_runner(main_worker, get_args())
