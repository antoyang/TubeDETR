# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim

import util.dist as dist
from datasets.vidstg_eval import VidSTGEvaluator
from datasets.hcstvg_eval import HCSTVGEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    writer=None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None
        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]

        targets = targets_to(targets, device)

        # forward
        memory_cache = model(
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
        )
        outputs = model(
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
        )

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        device = outputs["pred_boxes"].device
        inter_idx = batch_dict["inter_idx"]
        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            keep_list.extend(
                [
                    elt
                    for elt in range(
                        i_dur * max_duration + inter[0],
                        (i_dur * max_duration) + inter[1] + 1,
                    )
                ]
            )
        keep = torch.tensor(keep_list).long().to(device)
        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        for i_aux in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
                "pred_boxes"
            ][keep]
        b = len(durations)
        targets = [
            x for x in targets if len(x["boxes"])
        ]  # keep only targets in the annotated moment
        assert len(targets) == len(outputs["pred_boxes"]), (
            len(outputs["pred_boxes"]),
            len(targets),
        )
        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # compute losses
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, inter_idx, time_mask))

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if writer is not None and dist.is_main_process() and i % 100 == 0:
            for k in loss_dict_reduced_unscaled:
                writer.add_scalar(f"{k}", metric_logger.meters[k].avg, i)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None
        durations = batch_dict["durations"]
        captions = batch_dict["captions"]
        targets = batch_dict["targets"]

        targets = targets_to(targets, device)

        # forward
        memory_cache = model(
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
        )
        outputs = model(
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
        )

        # only keep box predictions in the annotated moment
        max_duration = max(durations)
        inter_idx = batch_dict["inter_idx"]
        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            if inter[0] >= 0:
                keep_list.extend(
                    [
                        elt
                        for elt in range(
                            i_dur * max_duration + inter[0],
                            (i_dur * max_duration) + inter[1] + 1,
                        )
                    ]
                )
        keep = torch.tensor(keep_list).long().to(outputs["pred_boxes"].device)
        if args.test:
            pred_boxes_all = outputs["pred_boxes"]
            targets_all = [x for x in targets]
        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        for i_aux in range(len(outputs["aux_outputs"])):
            outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
                "pred_boxes"
            ][keep]
        b = len(durations)
        targets = [x for x in targets if len(x["boxes"])]
        assert len(targets) == len(outputs["pred_boxes"]), (
            len(targets),
            len(outputs["pred_boxes"]),
        )
        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # compute losses
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, inter_idx, time_mask))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        # update evaluator
        # if args.test:
        # outputs["pred_boxes"] = pred_boxes_all
        if args.test:
            targets = targets_all
            outputs["pred_boxes"] = pred_boxes_all
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        vidstg_res = {} if "vidstg" in postprocessors.keys() else None
        vidstg_video_res = {} if "vidstg" in postprocessors.keys() else None
        hcstvg_res = {} if "hcstvg" in postprocessors.keys() else None
        hcstvg_video_res = {} if "hcstvg" in postprocessors.keys() else None
        if "vidstg" in postprocessors.keys():
            video_ids = batch_dict["video_ids"]
            frames_id = batch_dict["frames_id"]
            if args.sted:
                pred_steds = postprocessors["vidstg"](
                    outputs, frames_id, video_ids=video_ids, time_mask=time_mask
                )

            image_ids = [t["image_id"] for t in targets]
            for im_id, result in zip(image_ids, results):
                vidstg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

            qtypes = batch_dict["qtype"]
            assert len(set(video_ids)) == len(qtypes)
            if args.sted:
                assert len(pred_steds) == len(qtypes)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    vidstg_video_res[video_id] = {
                        "sted": pred_sted,
                        "qtype": qtypes[video_id],
                    }
            else:
                for video_id in video_ids:
                    vidstg_video_res[video_id] = {
                        "qtype": qtypes[video_id],
                    }
            res = {
                target["image_id"]: output for target, output in zip(targets, results)
            }
        elif "hcstvg" in postprocessors.keys():
            video_ids = batch_dict["video_ids"]
            frames_id = batch_dict["frames_id"]
            if args.sted:
                pred_steds = postprocessors["hcstvg"](
                    outputs, frames_id, video_ids=video_ids, time_mask=time_mask
                )
            image_ids = [t["image_id"] for t in targets]
            for im_id, result in zip(image_ids, results):
                hcstvg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

            if args.sted:
                assert len(set(video_ids)) == len(pred_steds)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    hcstvg_video_res[video_id] = {"sted": pred_sted}
            else:
                hcstvg_video_res[video_id] = {}
            res = {
                target["image_id"]: output for target, output in zip(targets, results)
            }
        else:
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, results)
            }

        for evaluator in evaluator_list:
            if isinstance(evaluator, VidSTGEvaluator):
                evaluator.update(vidstg_res)
                evaluator.video_update(vidstg_video_res)
                if args.test:
                    tsa_weights = [
                        outputs["aux_outputs"][i_aux]["weights"]
                        for i_aux in range(len(outputs["aux_outputs"]))
                    ]
                    tsa_weights.append(outputs["weights"])
                    weights = torch.stack(tsa_weights)
                    ca_weights = [
                        outputs["aux_outputs"][i_aux]["ca_weights"]
                        for i_aux in range(len(outputs["aux_outputs"]))
                    ]
                    ca_weights.append(outputs["ca_weights"])
                    ca_weights = torch.stack(ca_weights)
                    text_weights = ca_weights[
                        ..., -len(memory_cache["text_memory_resized"]) :
                    ]
                    spatial_weights = ca_weights[
                        ..., : -len(memory_cache["text_memory_resized"])
                    ].reshape(
                        ca_weights.shape[0],
                        ca_weights.shape[1],
                        ca_weights.shape[2],
                        math.ceil(samples.tensors.shape[2] / 32),
                        -1,
                    )  # hw
                    # tokens = memory_cache['tokenized'].tokens()
                    evaluator.save(
                        weights,
                        text_weights,
                        spatial_weights,
                        outputs["pred_sted"],
                        image_ids,
                        video_ids,
                    )
            elif isinstance(evaluator, HCSTVGEvaluator):
                evaluator.update(hcstvg_res)
                evaluator.video_update(hcstvg_video_res)
            else:
                evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    vidstg_res = None
    hcstvg_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, VidSTGEvaluator):
            vidstg_res = evaluator.summarize()
        elif isinstance(evaluator, HCSTVGEvaluator):
            hcstvg_res = evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if vidstg_res is not None:
        stats["vidstg"] = vidstg_res

    if hcstvg_res is not None:
        stats["hcstvg"] = hcstvg_res

    return stats
