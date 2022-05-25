from pathlib import Path
from typing import Dict, List

import numpy as np

import util.dist as dist

import json
from functools import reduce
from util.box_ops import np_box_iou


class HCSTVGGiouEvaluator:
    def __init__(
        self,
        hcstvg_path: str,
        subset: str = "test",
        verbose: bool = True,
        iou_thresholds: list = [0.3, 0.5],
        fps: int = 5,
        video_max_len: int = 200,
        v2=False,
        tmp_loc=True,
    ):
        """
        :param hcstvg_path: path to HC-STVG annotations
        :param subset: train, val or test
        :param verbose: whether to print more information or not
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param fps: number of frames per second
        :param video_max_len: maximum number of frames to be extracted from a video
        :param v2: whether to use the second version of the dataset
        :param tmp_loc: whether to evaluate temporal localization
        """

        assert subset in ["train", "val", "test"], f"Wrong HC-STVG subset {subset}"

        self.iou_thresholds = iou_thresholds
        self.tmp_loc = tmp_loc

        hcstvg_path = Path(hcstvg_path)

        # We load the image ids corresponding to the current subset
        if not v2:
            path = hcstvg_path / f"{subset}_proc.json"
        else:
            path = hcstvg_path / f"{subset}v2_proc.json"

        self.anns = json.load(open(path, "r"))
        self.vid2imgids = (
            {}
        )  # map video_id to list of corresponding frames to forward, index in all images of the first image to forward, and list of corresponding frames in the GT tube
        self.vid2steds = {}  # map video_id to [start, end] of the GT tube
        self.img2box = {}  # map video_id + frame_id to bbox
        for video in self.anns:
            video_num_images = video["frame_count"]
            video_id = video["video_id"]
            video_fps = video_num_images / 20
            sampling_rate = fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
            end_frame = (
                video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            )
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)
            if (
                len(frame_ids) > video_max_len
            ):  # temporal downsampling if there is still too many images
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]
            inter_frames = []
            self.vid2steds[video_id] = [
                video["tube_start_frame"],
                video["tube_end_frame"],
            ]
            for frame_id in frame_ids:
                if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                    x1, y1, w, h = video["trajectory"][
                        frame_id - video["tube_start_frame"]
                    ]
                    x2 = x1 + w
                    y2 = y1 + h
                    self.img2box[f"{video_id}_{frame_id}"] = [[x1, y1, x2, y2]]
                    inter_frames.append(f"{video_id}_{frame_id}")
            self.vid2imgids[video_id] = [frame_ids, inter_frames]

        if verbose:
            print(f"HC-STVG subset contains {len(self.vid2imgids)} videos")
            print(f"There are {len(self.imgid2box)} images to evaluate")

    def evaluate(self, predictions: List[Dict], video_predictions: List[Dict]):
        if len(video_predictions) < len(self.vid2imgids):
            raise RuntimeError(
                f"{len(self.vid2imgids) - len(video_predictions)} video predictions missing"
            )
        if len(predictions) < len(self.img2box):
            raise RuntimeError(
                f"{len(self.img2box) - len(predictions)} box predictions missing"
            )
        vid_metrics = {}
        for video_id, video_pred in video_predictions.items():
            if video_id in vid_metrics:
                print(f"Warning, multiple predictions found for video {video_id}")
                continue
            if self.tmp_loc:
                gt_sted = self.vid2steds[video_id]
                pred_sted = video_pred["sted"]
            frame_ids, inter_frames = self.vid2imgids[video_id]

            # compute temporal iou
            if self.tmp_loc:
                max_start = max(gt_sted[0], pred_sted[0])
                min_end = min(gt_sted[1], pred_sted[1])
                min_start = min(gt_sted[0], pred_sted[0])
                max_end = max(gt_sted[1], pred_sted[1])
                if min_end <= max_start:
                    tiou = 0
                else:
                    intersection = min_end - max_start
                    gt_span = gt_sted[1] - gt_sted[0]
                    pred_span = pred_sted[1] - pred_sted[0]
                    union = gt_span + pred_span - intersection
                    tiou = intersection / union

                # compute viou and gt_viou
                vid_metrics[video_id] = {
                    "gt_sted": gt_sted,
                    "pred_sted": pred_sted,
                    "tiou": tiou,
                    "img_metrics": {},
                }
                union_predgt = [
                    frame_id
                    for frame_id in frame_ids
                    if min_start <= frame_id < max_end
                ]
                inter_predgt = set(
                    [
                        frame_id
                        for frame_id in frame_ids
                        if max_start <= frame_id < min_end
                    ]
                )
                viou = 0
            else:
                vid_metrics[video_id] = {
                    "img_metrics": {},
                }
                union_predgt = frame_ids
                inter_predgt = frame_ids
            gt_viou = 0

            for (
                image_id
            ) in (
                inter_frames
            ):  # iterate on all frames of the annotated moment to update GT metrics
                if image_id not in predictions:
                    raise RuntimeError(f"No prediction for frame {image_id}")
                gt_boxes = self.img2box[image_id]
                pred_boxes = predictions[image_id]["boxes"]
                iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))[0][0]
                frame_id = int(image_id.split("_")[1])
                vid_metrics[video_id]["img_metrics"][image_id] = {
                    "iou": iou,
                    "pred_box": pred_boxes[0],
                    "gt_box": gt_boxes[0],
                }
                if (
                    frame_id in inter_predgt and self.tmp_loc
                ):  # update viou if this frame is in the intersection between the annotated moment and the predicted moment
                    viou += iou
                gt_viou += iou

            if self.tmp_loc:  # compute viou@R
                viou = viou / max(len(union_predgt), 1)
                vid_metrics[video_id]["viou"] = viou
                recalls = {thresh: 0 for thresh in self.iou_thresholds}
                for thresh in self.iou_thresholds:
                    if viou > thresh:
                        recalls[thresh] += 1

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(inter_frames), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            vid_metrics[video_id].update(
                {f"viou@{thresh}": recalls[thresh] for thresh in self.iou_thresholds}
            )
            vid_metrics[video_id].update(
                {
                    f"gt_viou@{thresh}": gt_recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

        return vid_metrics


class HCSTVGEvaluator(object):
    def __init__(
        self,
        hcstvg_path,
        subset,
        iou_thresholds=[0.3, 0.5],
        fps=5,
        video_max_len=200,
        v2=False,
        save_pred=False,
        tmp_loc=True,
    ):
        """
        :param hcstvg_path: path to HC-STVG annotations
        :param subset: train, val or test
        :param verbose: whether to print more information or not
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param fps: number of frames per second
        :param video_max_len: maximum number of frames to be extracted from a video
        :param v2: whether to use the second version of the dataset
        :param save_pred: whether to save predictions in the output of summarize
        """
        self.evaluator = HCSTVGGiouEvaluator(
            hcstvg_path,
            subset=subset,
            verbose=False,
            iou_thresholds=iou_thresholds,
            fps=fps,
            video_max_len=video_max_len,
            v2=v2,
            tmp_loc=tmp_loc,
        )
        self.predictions = {}
        self.video_predictions = {}
        self.results = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.tmp_loc = tmp_loc

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def video_update(self, video_predictions):
        self.video_predictions.update(video_predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        all_video_predictions = dist.all_gather(self.video_predictions)
        self.video_predictions = reduce(
            lambda a, b: a.update(b) or a, all_video_predictions, {}
        )

    def summarize(self):
        if dist.is_main_process():
            self.results = self.evaluator.evaluate(
                self.predictions, self.video_predictions
            )
            metrics = {"gt_viou": 0}
            if self.tmp_loc:
                metrics.update({"tiou": 0, "viou": 0})
            for thresh in self.iou_thresholds:  # init metrics
                if self.tmp_loc:
                    metrics[f"viou@{thresh}"] = 0
                metrics[f"gt_viou@{thresh}"] = 0
            counter = 0
            for x in self.results.values():  # sum results
                if self.tmp_loc:
                    metrics["tiou"] += x["tiou"]
                    metrics["viou"] += x["viou"]
                metrics["gt_viou"] += x["gt_viou"]
                for thresh in self.iou_thresholds:
                    if self.tmp_loc:
                        metrics[f"viou@{thresh}"] += x[f"viou@{thresh}"]
                    metrics[f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
                counter += 1
            for key in metrics:  # average results
                metrics[key] = metrics[key] / counter
                print(f"{key}: {metrics[key]:.4f}")
            out = {f"{name}": metrics[name] for name in metrics}
            if self.save_pred:
                out["predictions"] = self.predictions
                out["video_predictions"] = self.video_predictions
                out["vid_metrics"] = self.results
            return out

        return None, None
