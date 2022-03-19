# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform TubeDETR output according to the downstream task"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops


class PostProcessSTVG(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, frames_id=None, video_ids=None, time_mask=None):
        """
        :param outputs: must contain a key pred_sted mapped to a [B, T, 2] tensor of logits for the start and end predictions
        :param frames_id: list of B lists which contains the increasing list of frame ids corresponding to the indexes of the decoder outputs
        :param video_ids: list of B video_ids, used to ensemble predictions when video_max_len_train < video_max_len
        :param time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the possible predictions
        :return: list of B [start_frame, end_frame] for each video
        """
        steds = outputs["pred_sted"]  # BxTx2
        if len(set(video_ids)) != len(
            video_ids
        ):  # concatenate start and end probabilities predictions across all clips
            steds_list = [steds[0].masked_fill(~time_mask[0][:, None], -float("inf"))]
            for i_vid in range(1, len(video_ids)):
                if (
                    video_ids[i_vid] == video_ids[i_vid - 1]
                ):  # same video, concatenate prob logits
                    steds_list[-1] = torch.cat(
                        [
                            steds_list[-1],
                            steds[i_vid].masked_fill(
                                ~time_mask[i_vid][:, None], -float("inf")
                            ),
                        ],
                        0,
                    )
                else:  # new video
                    steds_list.append(
                        steds[i_vid].masked_fill(
                            ~time_mask[i_vid][:, None], -float("inf")
                        )
                    )
            n_videos = len(set(video_ids))
            max_dur = max(len(x) for x in steds_list)
            eff_steds = torch.ones(n_videos, max_dur, 2) * float("-inf")
            for i_v in range(len(steds_list)):
                eff_steds[i_v, : len(steds_list[i_v])] = steds_list[i_v]
            steds = eff_steds
        # put 0 probability to positions corresponding to end <= start
        mask = (
            (torch.ones(steds.shape[1], steds.shape[1]) * float("-inf"))
            .to(steds.device)
            .tril(0)
            .unsqueeze(0)
            .expand(steds.shape[0], -1, -1)
        )  # BxTxT
        starts_distribution = steds[:, :, 0].log_softmax(1)  # BxT
        ends_distribution = steds[:, :, 1].log_softmax(1)  # BxT
        # add log <=> multiply probs
        score = (
            starts_distribution.unsqueeze(2) + ends_distribution.unsqueeze(1)
        ) + mask  # BxTxT
        score, s_idx = score.max(dim=1)  # both BxT
        score, e_idx = score.max(dim=1)  # both B
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(1)  # B
        pred_steds = torch.stack([s_idx, e_idx], 1)  # Bx2
        # max_length = max([len(x) for x in frames_id])
        max_length = steds.shape[1]
        frames_id = (
            torch.tensor([row + [0] * (max_length - len(row)) for row in frames_id])
            .long()
            .to(pred_steds.device)
        )  # padded up to BxT
        # get corresponding frames id from the indexes
        pred_steds = torch.gather(frames_id, 1, pred_steds)
        pred_steds = pred_steds.float()
        pred_steds[:, 1] += 1  # the end frame is excluded in evaluation

        pred_steds = pred_steds.cpu().tolist()
        return pred_steds


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs["pred_boxes"]
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct

        results = [{"boxes": b} for b in boxes]

        return results


def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}

    if dataset_name in ["vidstg", "hcstvg"]:
        postprocessors[dataset_name] = PostProcessSTVG()

    return postprocessors
