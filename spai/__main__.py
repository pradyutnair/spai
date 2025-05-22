# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import neptune
import cv2
import click
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yacs
import filetype
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from yacs.config import CfgNode

import spai.data.data_finetune
from spai.config import get_config
from spai.models import build_cls_model
from spai.data import build_loader, build_loader_test
from spai.lr_scheduler import build_scheduler
from spai.models.sid import AttentionMask
from spai.models.semantic_fusion import AdaptiveSemanticSpectralFusion
from spai.onnx import compare_pytorch_onnx_models
from spai.optimizer import build_optimizer
from spai.logger import create_logger
from spai.utils import (
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    find_pretrained_checkpoints,
    inf_nan_to_num
)
from spai.models import losses
from spai import metrics
from spai import data_utils

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

cv2.setNumThreads(1)
logger: Optional[logging.Logger] = None


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--learning-rate", type=float)
@click.option("--data-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="path to dataset")
@click.option("--csv-root-dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--pretrained",
              type=click.Path(exists=True, dir_okay=False),
              help="path to pre-trained model")
@click.option("--resume", is_flag=True,
              help="resume from checkpoint")
@click.option("--accumulation-steps", type=int, default=1,
              help="Gradient accumulation steps.")
@click.option("--use-checkpoint", is_flag=True,
              help="Whether to use gradient checkpointing to save memory.")
@click.option("--amp-opt-level", type=click.Choice(["O0", "O1", "O2"]), default="O1",
              help="mixed precision opt level, if O0, no amp is used")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--local_rank", type=int, default=0,
              help="local_rank for distributed training")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--data-workers", type=int,
              help="Number of worker processes to be used for data loading.")
@click.option("--disable-pin-memory", is_flag=True)
@click.option("--data-prefetch-factor", type=int)
@click.option("--save-all", is_flag=True)
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def train(
    cfg: Path,
    batch_size: Optional[int],
    learning_rate: Optional[float],
    data_path: Path,
    csv_root_dir: Optional[Path],
    lmdb_path: Optional[Path],
    pretrained: Optional[Path],
    resume: bool,
    accumulation_steps: int,
    use_checkpoint: bool,
    amp_opt_level: str,
    output: Path,
    tag: str,
    local_rank: int,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    data_workers: Optional[int],
    disable_pin_memory: bool,
    data_prefetch_factor: Optional[int],
    save_all: bool,
    extra_options: tuple[str, str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = data_path.parent
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "data_path": str(data_path),
        "csv_root_dir": str(csv_root_dir),
        "lmdb_path": str(lmdb_path),
        "pretrained": str(pretrained) if pretrained is not None else None,
        "resume": resume,
        "accumulation_steps": accumulation_steps,
        "use_checkpoint": use_checkpoint,
        "amp_opt_level": amp_opt_level,
        "output": str(output),
        "tag": tag,
        "local_rank": local_rank,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "data_workers": data_workers,
        "disable_pin_memory": disable_pin_memory,
        "data_prefetch_factor": data_prefetch_factor,
        "opts": extra_options
    })
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # Set a fixed seed to all the random number generators.
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if config.TRAIN.SCALE_LR:
        # Linear scale the learning rate according to total batch size - may not be optimal.
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # Gradient accumulation also need to scale the learning rate.
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export and display current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config, logger, is_pretrain=False, is_test=False
    )

    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=["mfm", "train", config.TRAIN.MODE, data_path.stem]
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))

    # Step 1: Log initial parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Initial parameter counts:")
    logger.info(f"  • Total parameters: {total_params:,}")
    logger.info(f"  • Trainable parameters: {trainable_params:,}")
    logger.info(f"  • Frozen parameters: {total_params - trainable_params:,}")

    # Step 2: Freeze backbone parameters
    logger.info("🔒 Freezing backbone parameters...")
    for name, param in model.named_parameters():
        if any(x in name for x in ['vit.', 'backbone.']):
            param.requires_grad = False
            logger.info(f"  • Frozen: {name}")
        else:
            logger.info(f"  • Trainable: {name}")

    # Step 3: Log parameter counts after freezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Parameter counts after freezing:")
    logger.info(f"  • Total parameters: {total_params:,}")
    logger.info(f"  • Trainable parameters: {trainable_params:,}")
    logger.info(f"  • Frozen parameters: {total_params - trainable_params:,}")

    # Step 4: Group parameters by trainable status
    trainable_params_by_module = {}
    frozen_params_by_module = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if param.requires_grad:
            trainable_params_by_module[module_name] = trainable_params_by_module.get(module_name, 0) + param.numel()
        else:
            frozen_params_by_module[module_name] = frozen_params_by_module.get(module_name, 0) + param.numel()

    logger.info("📊 Parameter distribution by module:")
    logger.info("🔄 Trainable modules:")
    for module, count in trainable_params_by_module.items():
        logger.info(f"    • {module}: {count:,} parameters")
    logger.info("🔒 Frozen modules:")
    for module, count in frozen_params_by_module.items():
        logger.info(f"    • {module}: {count:,} parameters")

    # Step 5: Verify semantic components are trainable
    semantic_components = ['early_fusion_gates', 'cross_attention', 'cross_attention_norm']
    logger.info("🔍 Verifying semantic components:")
    for name, param in model.named_parameters():
        if any(comp in name for comp in semantic_components):
            # If they are initally frozen, they should be trainable
            if not param.requires_grad:
                param.requires_grad = True
                logger.info(f"  • {name} was frozen, now making it trainable: {param.requires_grad}")
            else:
                logger.info(f"  • {name} is already trainable: {param.requires_grad}")

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model_without_ddp = model

    # Step 6: Log final parameter counts after optimizer setup
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Final parameter counts after optimizer setup:")
    logger.info(f"  • Total parameters: {total_params:,}")
    logger.info(f"  • Trainable parameters: {trainable_params:,}")
    logger.info(f"  • Frozen parameters: {total_params - trainable_params:,}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion: nn.Module = losses.build_loss(config)
    logger.info(f"Loss: \n{criterion}")

    if config.PRETRAINED:
        model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]
        load_pretrained(config, model, logger, model_ckpt, verbose=True)
        logger.info("✅ Loaded pretrained weights from spai.pth")
    else:
        logger.info("⚠️ No pretrained model. Training from scratch.")

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)

    train_model(
        config,
        model,
        model_without_ddp,
        data_loader_train,
        data_loader_val,
        test_loaders,
        dataset_val,
        test_datasets,
        test_datasets_names,
        criterion,
        optimizer,
        lr_scheduler,
        log_writer,
        neptune_run,
        save_all=save_all
    )


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--update-csv", is_flag=True,
              help="When this flag is provided the predicted score for each sample is "
                   "written to the dataset csv, under a new column named as "
                   "{tag}_epoch_{epoch_num}_{crop_approach}.")
def test(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
    update_csv: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "test"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger,
                                                                         split=split)
    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    criterion = losses.build_loss(config)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        model.cuda()
        if i == 0:
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of Params: {n_parameters}")
            if hasattr(model, "flops"):
                flops = model.flops()
                logger.info(f"Number of GFLOPs: {flops / 1e9}")

        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i==0)

        # Test the model.
        for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                                  test_datasets,
                                                                  test_datasets_names):
            predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]] = None
            if update_csv:
                acc, ap, auc, loss, predictions = validate(
                    config, test_data_loader, model, criterion, neptune_run,
                    return_predictions=True
                )
            else:
                acc, ap, auc, loss = validate(config, test_data_loader,
                                              model, criterion, neptune_run)
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch} | "
                        f"Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | ACC: {acc:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | AP: {ap:.3f}")
            logger.info(f"Test | {test_data_name} | Epoch {checkpoint_epoch}  | "
                        f"Images: {len(test_dataset)} | AUC: {auc:.3f}")
            neptune_run[f"test/{test_data_name}/acc"].append(acc, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/ap"].append(ap, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/auc"].append(auc, step=checkpoint_epoch)
            neptune_run[f"test/{test_data_name}/loss"].append(loss, step=checkpoint_epoch)

            if predictions is not None:
                column_name: str = f"{tag}_epoch_{checkpoint_epoch}"
                scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
                attention_masks: dict[int, pathlib.Path] = {
                    i: t[1].mask for i, t in predictions.items() if t[1] is not None
                }
                test_dataset.update_dataset_csv(
                    column_name, scores, export_dir=Path(config.OUTPUT)
                )
                if len(attention_masks) == len(scores):
                    test_dataset.update_dataset_csv(
                        f"{column_name}_mask", attention_masks, export_dir=Path(config.OUTPUT)
                    )

        if log_writer is not None:
            log_writer.flush()
        if neptune_run is not None:
            neptune_run.sync()


@cli.command()
@click.option("--cfg", default="./configs/spai.yaml", show_default=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a configuration file for SPAI.")
@click.option("--batch-size", type=int, default=1, show_default=True,
              help="Inference batch size.")
@click.option("--input", "input_paths", multiple=True, required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Can be either a directory containing the images to be analyzed or a "
                   "CSV file describing the data. This CSV file should include at least a "
                   "column named `image` with the path to the images.")
@click.option("--input-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the input csv files. "
                   "If this option is omitted, the parent directory of each input csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the input csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "input csv file. In that case, the number of provided input csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model", default="./weights/spai.pth",
              type=click.Path(exists=True, path_type=Path),
              help="Path to the a weight file of SPAI.")
@click.option("--output", default="./output",
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory where a CSV file containing .")
@click.option("--tag", type=str, help="Tag of experiment", default="spai")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def infer(
    cfg: Path,
    batch_size: int,
    input_paths: list[Path],
    input_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in input_paths],
        "test_csv_root": [str(p) for p in input_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })

    output.mkdir(exist_ok=True, parents=True)

    # Create a console logger.
    global logger
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create dataloaders.
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split, dummy_csv_dir=output
    )

    # Load the trained weights' checkpoint.
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]
    criterion = losses.build_loss(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    load_pretrained(config, model, logger,  checkpoint_path=model_ckpt, verbose=False)

    # Infer predictions and compute performance metrics (only on csv inputs with ground-truths).
    for test_data_loader, test_dataset, test_data_name, input_path in zip(test_loaders,
                                                                          test_datasets,
                                                                          test_datasets_names,
                                                                          input_paths):
        predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]]
        acc, ap, auc, loss, predictions = validate(
            config, test_data_loader, model, criterion, None, return_predictions=True
        )

        if input_path.is_file():  # When input path is a dir, no ground-truth exists.
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | ACC: {acc:.3f}")
            if test_dataset.get_classes_num() > 1:  # AUC and AP make no sense with only 1 class.
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AP: {ap:.3f}")
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AUC: {auc:.3f}")

        # Update the output CSV.
        if predictions is not None:
            column_name: str = f"{tag}"
            scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
            attention_masks: dict[int, pathlib.Path] = {
                i: t[1].mask for i, t in predictions.items() if t[1] is not None
            }
            test_dataset.update_dataset_csv(
                column_name, scores, export_dir=Path(output)
            )
            if len(attention_masks) == len(scores):
                test_dataset.update_dataset_csv(
                    f"{column_name}_mask", attention_masks, export_dir=Path(output)
                )


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--resize-to", type=int,
              help="When this argument is provided the testing images will be resized "
                   "so that their biggest dimension does not exceed this value.")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def tsne(
    cfg: Path,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    resize_to: Optional[int],
    extra_options: tuple[str, str],
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": 1,  # Currently, required to be 1 for correctly distinguishing embeddings.
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    from spai import tsne as tsne_utils

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "tsne"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of Params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    checkpoint_epoch: int = load_pretrained(config, model, logger, checkpoint_path=model_ckpt)

    # Test the model.
    for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                              test_datasets,
                                                              test_datasets_names):
        tsne_utils.visualize_tsne(config, test_data_loader, test_data_name, model, neptune_run)

        if log_writer is not None:
            log_writer.flush()
        if neptune_run is not None:
            neptune_run.sync()


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--exclude-preprocessing", is_flag=True,
              help="When this flag is provided the exported encoder does not include the spectral "
                   "filtering and normalization preprocessing operations. Instead, it accepts "
                   "three inputs, requiring these operations to be previously performed.")
def export_onnx(
    cfg: Path,
    model: Path,
    output: Path,
    tag: str,
    extra_options: tuple[str, str],
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    onnx_export_dir.mkdir(exist_ok=True, parents=True)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)

        model.to("cpu")
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"
        model.export_onnx(patch_encoder, patch_aggregator,
                          include_fft_preprocessing=not exclude_preprocessing)


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int, help="Batch size.")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--split", type=str, default="test",
              help="The data split which will be tested. Actually, this value is expected to be "
                   "present in the `split` column of the provided csv files. Only samples "
                   "in the csv belonging to the provided split will be tested.")
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--model",
              type=click.Path(exists=True),
              help="path to pre-trained model")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str, help="tag of experiment")
@click.option("--device", type=str, default="cpu")
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
@click.option("--exclude-preprocessing", is_flag=True)
def validate_onnx(
    cfg: Path,
    batch_size: Optional[int],
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    split: str,
    lmdb_path: Optional[Path],
    model: Path,
    output: Path,
    tag: str,
    device: str,
    extra_options: tuple[str, str],
    exclude_preprocessing: bool
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split
    )

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    if not onnx_export_dir.exists():
        raise FileNotFoundError(f"No onnx model at {onnx_export_dir}. Use the export-onnx "
                                f"command first.")

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)
        model.to(device)
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"

        compare_pytorch_onnx_models(
            model,
            patch_encoder,
            patch_aggregator,
            includes_preprocessing=not exclude_preprocessing,
            device=device
        )


def train_model(
    config: yacs.config.CfgNode,
    model: nn.Module,
    model_without_ddp: nn.Module,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_val: torch.utils.data.DataLoader,
    data_loaders_test: list[torch.utils.data.DataLoader],
    dataset_val: spai.data.data_finetune.CSVDataset,
    datasets_test: list[spai.data.data_finetune.CSVDataset],
    datasets_test_names: list[str],
    criterion,
    optimizer,
    lr_scheduler,
    log_writer,
    neptune_run,
    save_all: bool = False
) -> None:
    logger.info("🚀 Starting training")
    
    # Log training configuration settings
    if config.TRAIN.HIGH_RES_TRAINING:
        logger.info(f"📈 High-resolution training enabled: upscaling images to {config.TRAIN.HIGH_RES_SIZE}x{config.TRAIN.HIGH_RES_SIZE}")
        neptune_run["train/high_res_enabled"] = True
        neptune_run["train/high_res_size"] = config.TRAIN.HIGH_RES_SIZE
    else:
        logger.info(f"📋 Standard resolution training: images size {config.DATA.IMG_SIZE}x{config.DATA.IMG_SIZE}")
        neptune_run["train/high_res_enabled"] = False

    start_time: float = time.time()
    val_accuracy_per_epoch: list[float] = []
    val_ap_per_epoch: list[float] = []
    val_auc_per_epoch: list[float] = []
    val_loss_per_epoch: list[float] = []

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        epoch_start_time: float = time.time()

        # Stage 1: Train semantic components (first 5 epochs)
        if epoch < 5:
            logger.info(f"🎯 Stage 1: Training adaptive fusion components (Epoch {epoch})")
            
            # Focus on training the semantic fusion module by setting larger learning rate
            if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion') and \
               isinstance(model.mfvit.semantic_fusion, AdaptiveSemanticSpectralFusion):
                
                # Boost learning rate for fusion module parameters
                for param_group in optimizer.param_groups:
                    if any(name in param_group.get('names', []) for name in ['semantic_fusion', 'early_fusion_gates', 'frequency_band']):
                        original_lr = param_group['lr']
                        param_group['lr'] = original_lr * 2.0  # Boost learning rate for fusion parameters
                        logger.info(f"🔺 Increasing learning rate for fusion parameters: {original_lr:.6f} -> {param_group['lr']:.6f}")
                
                # Verify trainable status of key components
                fusion_module = model.mfvit.semantic_fusion
                logger.info(f"✅ Trainable status of fusion components:")
                logger.info(f"  • frequency_band_weights: {fusion_module.frequency_band_weights.requires_grad}")
                logger.info(f"  • early_fusion_gates: {[p.requires_grad for p in fusion_module.early_fusion_gates.parameters()]}")
                logger.info(f"  • modality_contribution: {fusion_module.modality_contribution.requires_grad}")
            
            train_one_epoch(
                config,
                model,
                criterion,
                data_loader_train,
                optimizer,
                epoch,
                lr_scheduler,
                log_writer,
                neptune_run,
                stage="semantic"
            )
            
            # Restore original learning rates after training
            if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion'):
                # Reset any boosted learning rates
                for param_group in optimizer.param_groups:
                    if 'original_lr' in param_group:
                        param_group['lr'] = param_group['original_lr']
                        logger.info(f"🔽 Reset learning rate to original: {param_group['lr']:.6f}")
                
        # Stage 2: Joint training (after 5 epochs)
        else:
            logger.info(f"🔄 Stage 2: Joint training (Epoch {epoch})")
            train_one_epoch(
                config,
                model,
                criterion,
                data_loader_train,
                optimizer,
                epoch,
                lr_scheduler,
                log_writer,
                neptune_run,
                stage="joint"
            )

        neptune_run["train/last_epoch"] = epoch + 1
        neptune_run["train/epochs_trained"] = epoch + 1 - config.TRAIN.START_EPOCH

        # Validate the model
        logger.info("📊 Validating model...")
        
        # Export attention visualizations on a regular basis (every 5 epochs)
        export_attn = (epoch % 5 == 0)
        if export_attn:
            logger.info("📷 Exporting attention visualizations during validation")
            
        acc, ap, auc, loss = validate(
            config, data_loader_val, model, criterion, neptune_run, 
            export_attention=export_attn
        )
        
        logger.info(f"✅ Val | Epoch {epoch} | Images: {len(dataset_val)} | loss: {loss:.4f}")
        logger.info(f"✅ Val | Epoch {epoch} | Images: {len(dataset_val)} | ACC: {acc:.3f}")
        logger.info(f"✅ Val | Epoch {epoch} | Images: {len(dataset_val)} | AP: {ap:.3f}")
        logger.info(f"✅ Val | Epoch {epoch} | Images: {len(dataset_val)} | AUC: {auc:.3f}")
        
        # Log metrics to Neptune
        neptune_run["val/auc"].append(auc)
        neptune_run["val/ap"].append(ap)
        neptune_run["val/accuracy"].append(acc)
        neptune_run["val/loss"].append(loss)

        # Display best epochs
        val_accuracy_per_epoch.append(acc)
        val_ap_per_epoch.append(ap)
        val_auc_per_epoch.append(auc)
        val_loss_per_epoch.append(loss)
        logger.info(f"🏆 Val | Min loss: {min(val_loss_per_epoch):.4f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmin(val_loss_per_epoch)}")
        logger.info(f"🏆 Val | Max ACC: {max(val_accuracy_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH+np.argmax(val_accuracy_per_epoch)}")
        logger.info(f"🏆 Val | Max AP: {max(val_ap_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_ap_per_epoch)}")
        logger.info(f"🏆 Val | Max AUC: {max(val_auc_per_epoch):.3f} "
                    f"| Epoch: {config.TRAIN.START_EPOCH + np.argmax(val_auc_per_epoch)}")

        # Save checkpoints
        if len(val_loss_per_epoch) == 1 or loss < min(val_loss_per_epoch[:-1]) or save_all:
            logger.info("💾 Saving checkpoint...")
            save_checkpoint(config, epoch, model_without_ddp, max(val_accuracy_per_epoch),
                            optimizer, lr_scheduler, logger)

        # Test the model
        for test_data_loader, test_dataset, test_data_name in zip(data_loaders_test,
                                                                  datasets_test,
                                                                  datasets_test_names):
            logger.info(f"🧪 Testing on {test_data_name}...")
            
            # Export attention visualizations on the same schedule as validation
            acc, ap, auc, loss = validate(
                config, test_data_loader, model, criterion, neptune_run,
                export_attention=export_attn
            )
            
            logger.info(f"📊 Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| loss: {loss:.4f}")
            logger.info(f"📊 Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| ACC: {acc:.3f}")
            logger.info(f"📊 Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AP: {ap:.3f}")
            logger.info(f"📊 Test | {test_data_name} | Epoch {epoch} | Images: {len(test_dataset)} "
                        f"| AUC: {auc:.3f}")
            
            # Log test metrics
            neptune_run[f"test/{test_data_name}/acc"].append(acc)
            neptune_run[f"test/{test_data_name}/ap"].append(ap)
            neptune_run[f"test/{test_data_name}/auc"].append(auc)
            neptune_run[f"test/{test_data_name}/loss"].append(loss if not np.isnan(loss) else -100.)

        # Log epoch time
        epoch_time: float = time.time() - epoch_start_time
        logger.info(f"⏱️ Epoch training time: {epoch_time:.3f}s")
        neptune_run["train/epoch_train_time"].append(epoch_time)

        if neptune_run is not None:
            neptune_run.sync()

    # Log total training time
    total_time: float = time.time() - start_time
    total_time_str: str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"⏱️ Overall training time: {total_time_str}")
    neptune_run["train/total_train_time"].append(total_time_str)


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    log_writer,
    neptune_run,
    stage: str = "joint"
):
    model.train()
    criterion.train()
    optimizer.zero_grad()
    
    logger.info(
        "📈 Current learning rate for different parameter groups: "
        f"{[it['lr'] for it in optimizer.param_groups]}"
    )

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    # Initialize metrics for fusion module tracking
    semantic_weight_meter = AverageMeter()
    spectral_weight_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    # Adjust memory handling for high-resolution training
    if config.TRAIN.HIGH_RES_TRAINING:
        torch.cuda.empty_cache()  # Clear GPU cache before starting the epoch
        logger.info(f"🧹 Cleared GPU cache for high-resolution training ({config.TRAIN.HIGH_RES_SIZE}x{config.TRAIN.HIGH_RES_SIZE})")
    
    for idx, batch in enumerate(data_loader):
        if isinstance(criterion, TripletMarginLoss):
            anchor, positive, negative = batch
            batch_size: int = anchor.size(0)
            anchor = anchor.cuda(non_blocking=True)
            positive = positive.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
        else:
            samples, targets, _ = batch
            batch_size: int = samples.size(0)
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # Forward pass each augmented view of the batch separately
            outputs_views: list[torch.Tensor] = []
            for i in range(samples.size(1)):
                # High-resolution handling - use smaller batches if necessary
                view = samples[:, i, :, :, :].float()
                
                # For logging attention weights (only do this periodically to save time)
                if idx % config.PRINT_FREQ == 0 and i == 0 and neptune_run is not None and \
                   hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion') and \
                   isinstance(model.mfvit.semantic_fusion, AdaptiveSemanticSpectralFusion):
                    # Export attention for the first view in the batch
                    attention_dir = pathlib.Path(f"attention_vis/epoch_{epoch}_batch_{idx}")
                    output = model(view, export_attention=True)
                    
                    # Get attention weights if available
                    fusion_module = model.mfvit.semantic_fusion
                    attention_data = fusion_module.get_last_attention_weights()
                    
                    if attention_data.get('modality_contribution') is not None:
                        mod_weights = attention_data['modality_contribution'].numpy()
                        semantic_weight_meter.update(mod_weights[0], 1)  # Semantic weight
                        spectral_weight_meter.update(mod_weights[1], 1)  # Spectral weight
                        
                        # Log to Neptune
                        neptune_run[f"fusion/semantic_weight"].append(mod_weights[0])
                        neptune_run[f"fusion/spectral_weight"].append(mod_weights[1])
                        
                    if attention_data.get('frequency_weights') is not None:
                        freq_weights = attention_data['frequency_weights'].numpy()
                        # Log to Neptune
                        for i, w in enumerate(freq_weights):
                            neptune_run[f"fusion/freq_band_{i}_weight"].append(w)
                            
                    # Log attention map
                    if attention_data.get('attention_map') is not None:
                        import matplotlib.pyplot as plt
                        import io
                        from PIL import Image
                        
                        # Create and save attention map figure
                        attn_map = attention_data['attention_map'].numpy()
                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(attn_map, cmap='viridis')
                        plt.colorbar(im)
                        plt.title(f"Attention Map (Epoch {epoch})")
                        
                        # Save to buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        
                        # Log to Neptune
                        neptune_run[f"fusion/attention_map"].upload(Image.open(buf))
                        plt.close(fig)
                else:
                    # Normal forward pass
                    try:
                        output = model(view)
                    except RuntimeError as e:
                        if "out of memory" in str(e) and config.TRAIN.HIGH_RES_TRAINING:
                            # Handle OOM errors during high-res training by clearing cache and reducing batch
                            logger.warning(f"⚠️ OOM error at batch {idx} - clearing cache and retrying with reduced batch")
                            torch.cuda.empty_cache()
                            # Try again with half the batch
                            half_batch = view.shape[0] // 2
                            if half_batch >= 1:
                                outputs1 = model(view[:half_batch])
                                torch.cuda.empty_cache()
                                outputs2 = model(view[half_batch:])
                                output = torch.cat([outputs1, outputs2], dim=0)
                            else:
                                # Can't split batch further, skip this batch
                                logger.warning(f"⚠️ Skipping batch {idx} - cannot split further")
                                break
                        else:
                            raise e
                    
                outputs_views.append(output)
            
            # Skip processing if batch was skipped due to OOM
            if len(outputs_views) == 0:
                continue
                
            outputs: torch.Tensor = torch.stack(outputs_views, dim=1)
            outputs = outputs if outputs.size(dim=1) > 1 else outputs.squeeze(dim=1)

        # Compute loss
        if isinstance(criterion, TripletMarginLoss):
            loss = criterion(anchor_outputs, positive_outputs, negative_outputs)
        else:
            loss = criterion(outputs.squeeze(), targets)
            
        # Backward pass
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'⏱️ Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {optimizer.param_groups[-1]["lr"]:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
            # Log fusion module stats if available
            if semantic_weight_meter.count > 0:
                logger.info(
                    f'🔀 Fusion: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'semantic weight {semantic_weight_meter.avg:.4f}\t'
                    f'spectral weight {spectral_weight_meter.avg:.4f}'
                )

    epoch_time = time.time() - start
    logger.info(f"⏱️ EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    # Log epoch-level fusion stats to Neptune
    if semantic_weight_meter.count > 0 and neptune_run is not None:
        neptune_run[f"fusion/epoch_semantic_weight"].append(semantic_weight_meter.avg, step=epoch)
        neptune_run[f"fusion/epoch_spectral_weight"].append(spectral_weight_meter.avg, step=epoch)


@torch.no_grad()
def validate(
    config,
    data_loader,
    model,
    criterion,
    neptune_run,
    verbose: bool = True,
    return_predictions: bool = False,
    export_attention: bool = False
):
    model.eval()
    criterion.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_metrics: metrics.Metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    predicted_scores: dict[int, tuple[float, Optional[AttentionMask]]] = {}
    
    # Initialize metrics for fusion module tracking
    semantic_weight_meter = AverageMeter()
    spectral_weight_meter = AverageMeter()

    # Clear cache for high-res validation
    if hasattr(config.TRAIN, 'HIGH_RES_TRAINING') and config.TRAIN.HIGH_RES_TRAINING:
        torch.cuda.empty_cache()
        if verbose:
            logger.info(f"🧹 Cleared GPU cache for high-resolution validation")

    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            # In case of arbitrary resolution models the batch is provided as a list of tensors.
            images = [img.cuda(non_blocking=True) for img in images]
            # Remove views dimension. Always 1 during inference.
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Compute output.
        try:
            if isinstance(images, list) and config.TEST.EXPORT_IMAGE_PATCHES:
                export_dirs: list[pathlib.Path] = [
                    pathlib.Path(config.OUTPUT)/"images"/f"{dataset_idx.detach().cpu().tolist()[i]}"
                    for i in range(len(dataset_idx))
                ]
                
                # Set up attention export directories if needed
                attention_dirs = None
                if export_attention:
                    attention_dirs = [
                        dir_path / "attention" for dir_path in export_dirs
                    ]
                    for attn_dir in attention_dirs:
                        attn_dir.mkdir(exist_ok=True, parents=True)
                        
                output, attention_masks = model(
                    images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_dirs, 
                    export_attention=export_attention
                )
            elif isinstance(images, list):
                output = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_attention=export_attention)
                attention_masks = [None] * len(images)
            else:
                if images.size(dim=1) > 1:
                    predictions: list[torch.Tensor] = []
                    
                    for i in range(images.size(dim=1)):
                        # Configure attention directory if needed
                        if export_attention and i == 0 and idx % config.PRINT_FREQ == 0:
                            attention_dir = pathlib.Path(config.OUTPUT) / f"attention_val/batch_{idx}_view_{i}"
                            prediction = model(images[:, i], export_attention=True)
                            
                            # Log fusion module stats if available
                            if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion') and \
                               isinstance(model.mfvit.semantic_fusion, AdaptiveSemanticSpectralFusion) and \
                               neptune_run is not None:
                                
                                fusion_module = model.mfvit.semantic_fusion
                                attention_data = fusion_module.get_last_attention_weights()
                                
                                if attention_data.get('modality_contribution') is not None:
                                    mod_weights = attention_data['modality_contribution'].numpy()
                                    semantic_weight_meter.update(mod_weights[0], 1)
                                    spectral_weight_meter.update(mod_weights[1], 1)
                                    
                                    # Log to Neptune
                                    neptune_run[f"val/fusion/semantic_weight"].append(mod_weights[0])
                                    neptune_run[f"val/fusion/spectral_weight"].append(mod_weights[1])
                        else:
                            prediction = model(images[:, i])
                            
                        predictions.append(prediction)
                        
                    predictions: torch.Tensor = torch.stack(predictions, dim=1)
                    if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
                        output: torch.Tensor = predictions.max(dim=1).values
                    elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
                        output: torch.Tensor = predictions.mean(dim=1)
                    else:
                        raise TypeError(f"{config.TEST.VIEWS_REDUCTION_APPROACH} is not a "
                                        f"supported views reduction approach")
                else:
                    images = images.squeeze(dim=1)  # Remove views dimension.
                    
                    # Configure attention directory if needed
                    if export_attention and idx % config.PRINT_FREQ == 0:
                        attention_dir = pathlib.Path(config.OUTPUT) / f"attention_val/batch_{idx}"
                        output = model(images, export_attention=True)
                        
                        # Log fusion module stats if available
                        if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion') and \
                           isinstance(model.mfvit.semantic_fusion, AdaptiveSemanticSpectralFusion) and \
                           neptune_run is not None:
                            
                            fusion_module = model.mfvit.semantic_fusion
                            attention_data = fusion_module.get_last_attention_weights()
                            
                            if attention_data.get('modality_contribution') is not None:
                                mod_weights = attention_data['modality_contribution'].numpy()
                                semantic_weight_meter.update(mod_weights[0], 1)
                                spectral_weight_meter.update(mod_weights[1], 1)
                                
                                # Log to Neptune
                                neptune_run[f"val/fusion/semantic_weight"].append(mod_weights[0])
                                neptune_run[f"val/fusion/spectral_weight"].append(mod_weights[1])
                    else:
                        output = model(images)
                        
                attention_masks = [None] * images.size(0)
                
        except RuntimeError as e:
            if "out of memory" in str(e) and hasattr(config.TRAIN, 'HIGH_RES_TRAINING') and config.TRAIN.HIGH_RES_TRAINING:
                # Handle OOM during validation with high-res images
                logger.warning(f"⚠️ OOM error during validation at batch {idx} - skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        loss = criterion(output.squeeze(dim=1), target)

        # Apply sigmoid to output.
        output = torch.sigmoid(output)

        # Update metrics.
        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(output[:, 0].cpu(), target.cpu())

        # Keep predictions if requested.
        if return_predictions:
            batch_predictions: list[float] = output.squeeze(dim=1).detach().cpu().tolist()
            batch_dataset_idx: list[int] = dataset_idx.detach().cpu().tolist()
            predicted_scores.update({
                i: (p, m) for i, p, m in zip(batch_dataset_idx, batch_predictions, attention_masks)
            })

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and verbose:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}] | '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'Mem {memory_used:.0f}MB')
                
            # Log fusion module stats if available
            if semantic_weight_meter.count > 0 and verbose:
                logger.info(
                    f'🔀 Val Fusion: [{idx}/{len(data_loader)}] | '
                    f'semantic weight {semantic_weight_meter.avg:.4f} | '
                    f'spectral weight {spectral_weight_meter.avg:.4f}'
                )

    metric_values: dict[str, np.ndarray] = cls_metrics.compute()
    auc: float = metric_values["auc"].item()
    ap: float = metric_values["ap"].item()
    acc: float = metric_values["accuracy"].item()
    
    # Log fusion stats to Neptune at the end of validation
    if semantic_weight_meter.count > 0 and neptune_run is not None:
        neptune_run[f"val/fusion/avg_semantic_weight"].append(semantic_weight_meter.avg)
        neptune_run[f"val/fusion/avg_spectral_weight"].append(spectral_weight_meter.avg)

    if return_predictions:
        return acc, ap, auc, loss_meter.avg, predicted_scores
    else:
        return acc, ap, auc, loss_meter.avg


if __name__ == '__main__':
    cli()
