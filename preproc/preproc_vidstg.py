import json
import os
import copy
from tqdm import tqdm

# load config
with open("../config/vidstg.json", "r") as f:
    cfg = json.load(f)
video_path = os.path.join(cfg.vidstg_img_path, "video")
ann_path = cfg.vidstg_ann_path
vidor_path = cfg.vidstg_vid_path

# preproc VidOR annotations
vidor_dirs = ["training", "validation"]
categories = {}  # object categories, just for information
for dir in vidor_dirs:
    outs = {}  # maps video_id to trajectories
    subdirs = os.listdir(os.path.join(vidor_path, dir))
    for subdir in tqdm(subdirs):
        files = os.listdir(os.path.join(vidor_path, dir, subdir))
        for file in files:
            annot = json.load(open(os.path.join(vidor_path, dir, subdir, file), "r"))
            out = {
                "video_id": annot["video_id"],
                "video_path": annot["video_path"],
                "frame_count": annot["frame_count"],
                "fps": annot["fps"],
                "width": annot["width"],
                "height": annot["height"],
            }
            assert len(set(x["tid"] for x in annot["subject/objects"])) == len(
                annot["subject/objects"]
            )
            out["objects"] = {
                obj["tid"]: obj["category"] for obj in annot["subject/objects"]
            }
            trajectories = {}
            for i_frame, traj in enumerate(annot["trajectories"]):
                for bbox in traj:
                    if bbox["tid"] not in trajectories:
                        trajectories[bbox["tid"]] = {}
                        category = out["objects"][bbox["tid"]]
                        if category not in categories:
                            category_id = len(categories)
                            categories[category] = category_id
                        else:
                            category_id = categories[category]
                    trajectories[bbox["tid"]][i_frame] = {
                        "bbox": [
                            bbox["bbox"]["xmin"],
                            bbox["bbox"]["ymin"],
                            bbox["bbox"]["xmax"] - bbox["bbox"]["xmin"],
                            bbox["bbox"]["ymax"] - bbox["bbox"]["ymin"],
                        ],
                        "generated": bbox["generated"],
                        "tracker": bbox["tracker"],
                        "category_id": category_id,
                    }
            out["trajectories"] = trajectories
            outs[annot["video_id"]] = out
    json.dump(outs, open(os.path.join(ann_path, "vidor_" + dir + ".json"), "w"))
print(categories)

# preproc VidSTG annotations
files = ["train_annotations.json", "val_annotations.json", "test_annotations.json"]
for file in files:
    videos = []
    trajectories = {}
    annotations = json.load(open(os.path.join(ann_path, file), "r"))
    vidor = (
        json.load(open(os.path.join(ann_path, "vidor_training.json"), "r"))
        if "train" in file or "val" in file
        else json.load(open(os.path.join(ann_path, "vidor_validation.json"), "r"))
    )
    for annot in tqdm(annotations):
        annot_vidor = vidor[annot["vid"]]
        out = {
            "original_video_id": annot["vid"],
            "frame_count": annot["frame_count"],
            "fps": annot["fps"],
            "width": annot["width"],
            "height": annot["height"],
            "start_frame": annot["used_segment"]["begin_fid"],
            "end_frame": annot["used_segment"]["end_fid"],
            "tube_start_frame": annot["temporal_gt"]["begin_fid"],
            "tube_end_frame": annot["temporal_gt"]["end_fid"],
            "video_path": annot_vidor["video_path"],
        }

        for query in annot["questions"]:  # interrogative sentences
            video = copy.deepcopy(out)
            video["caption"] = query["description"]
            video["type"] = query["type"]
            video["target_id"] = query["target_id"]
            video["video_id"] = len(videos)
            video["qtype"] = "interrogative"
            videos.append(video)
            # get the trajectory
            if annot["vid"] not in trajectories:
                trajectories[annot["vid"]] = {}
            if str(query["target_id"]) not in trajectories[annot["vid"]]:
                trajectories[annot["vid"]][str(query["target_id"])] = annot_vidor[
                    "trajectories"
                ][str(query["target_id"])]
            # check that the annotated moment corresponds to annotated boxes
            assert annot["temporal_gt"]["end_fid"] - 1 <= max(
                int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
            ), (
                annot["temporal_gt"]["end_fid"],
                max(
                    int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
                ),
            )
            assert annot["temporal_gt"]["begin_fid"] >= min(
                int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
            ), (
                annot["temporal_gt"]["begin_fid"],
                min(
                    int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
                ),
            )

        for query in annot["captions"]:  # declarative sentences
            video = copy.deepcopy(out)
            video["caption"] = query["description"]
            video["type"] = query["type"]
            video["target_id"] = query["target_id"]
            video["video_id"] = len(videos)
            video["qtype"] = "declarative"
            videos.append(video)
            # get the trajectory
            if annot["vid"] not in trajectories:
                trajectories[annot["vid"]] = {}
            if str(query["target_id"]) not in trajectories[annot["vid"]]:
                trajectories[annot["vid"]][str(query["target_id"])] = annot_vidor[
                    "trajectories"
                ][str(query["target_id"])]
            # check that the annotated moment corresponds to annotated boxes
            assert annot["temporal_gt"]["end_fid"] - 1 <= max(
                int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
            ), (
                annot["temporal_gt"]["end_fid"],
                max(
                    int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
                ),
            )
            assert annot["temporal_gt"]["begin_fid"] >= min(
                int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
            ), (
                annot["temporal_gt"]["begin_fid"],
                min(
                    int(x) for x in trajectories[annot["vid"]][str(query["target_id"])]
                ),
            )
    outfile = {"videos": videos, "trajectories": trajectories}
    json.dump(outfile, open(os.path.join(ann_path, file.split("_")[0] + ".json"), "w"))
