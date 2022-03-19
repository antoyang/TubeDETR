import torch
import os
import numpy as np
import ffmpeg
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from main import get_args_parser
from models.transformer import build_transformer
from models.backbone import build_backbone
from models.tubedetr import TubeDETR
from models.postprocessors import PostProcessSTVG, PostProcess
from datasets.video_transforms import prepare, make_video_transforms
from util.misc import NestedTensor

parser = argparse.ArgumentParser(
    "TubeDETR training and evaluation script", parents=[get_args_parser()]
)
args = parser.parse_args()
device = args.device

# load model

backbone = build_backbone(args)

transformer = build_transformer(args)

model = TubeDETR(
    backbone,
    transformer,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
    video_max_len=args.video_max_len_train,
    stride=args.stride,
    guided_attn=args.guided_attn,
    fast=args.fast,
    fast_mode=args.fast_mode,
    sted=args.sted,
)
model.to(device)
print("model loaded")

postprocessors = {"vidstg": PostProcessSTVG(), "bbox": PostProcess()}

# load checkpoint
assert args.load
checkpoint = torch.load(args.load, map_location="cpu")
if "model_ema" in checkpoint:
    if (
        args.num_queries < 100 and "query_embed.weight" in checkpoint["model_ema"]
    ):  # initialize from the first object queries
        checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"][
            "query_embed.weight"
        ][: args.num_queries]
    if "transformer.time_embed.te" in checkpoint["model_ema"]:
        del checkpoint["model_ema"]["transformer.time_embed.te"]
    model.load_state_dict(checkpoint["model_ema"], strict=False)
else:
    if (
        args.num_queries < 100 and "query_embed.weight" in checkpoint["model"]
    ):  # initialize from the first object queries
        checkpoint["model"]["query_embed.weight"] = checkpoint["model"][
            "query_embed.weight"
        ][: args.num_queries]
    if "transformer.time_embed.te" in checkpoint["model"]:
        del checkpoint["model"]["transformer.time_embed.te"]
    model.load_state_dict(checkpoint["model"], strict=False)
print("checkpoint loaded")

# load video (with eventual start & end) & caption demo examples
captions = [args.caption_example]
vid_path = args.video_example
# get video metadata
probe = ffmpeg.probe(vid_path)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
)
"""num, denum = video_stream["avg_frame_rate"].split("/")
video_fps = int(num) / int(denum)"""
clip_start = (
    args.start_example if args.start_example >= 0 else float(video_stream["start_time"])
)
clip_end = (
    args.end_example
    if args.end_example > 0
    else float(video_stream["start_time"]) + float(video_stream["duration"])
)
ss = clip_start
t = clip_end - clip_start
extracted_fps = (
    min((args.fps * t), args.video_max_len) / t
)  # actual fps used for extraction given that the model processes video_max_len frames maximum
cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=extracted_fps)
out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
    capture_stdout=True, quiet=True
)
w = int(video_stream["width"])
h = int(video_stream["height"])
images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
assert len(images_list) <= args.video_max_len
image_ids = [[k for k in range(len(images_list))]]

# video transforms
empty_anns = []  # empty targets as placeholders for the transforms
placeholder_target = prepare(w, h, empty_anns)
placeholder_targets_list = [placeholder_target] * len(images_list)
transforms = make_video_transforms("test", cautious=True, resolution=args.resolution)
images, targets = transforms(images_list, placeholder_targets_list)
samples = NestedTensor.from_tensor_list([images], False)
if args.stride:
    samples_fast = samples.to(device)
    samples = NestedTensor.from_tensor_list([images[:, :: args.stride]], False).to(
        device
    )
else:
    samples_fast = None
durations = [len(targets)]

with torch.no_grad():  # forward through the model
    # encoder
    memory_cache = model(
        samples,
        durations,
        captions,
        encode_and_save=True,
        samples_fast=samples_fast,
    )
    # decoder
    outputs = model(
        samples,
        durations,
        captions,
        encode_and_save=False,
        memory_cache=memory_cache,
    )

    pred_steds = postprocessors["vidstg"](outputs, image_ids, video_ids=[0])[
        0
    ]  # (start, end) in terms of image_ids
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessors["bbox"](outputs, orig_target_sizes)
    vidstg_res = {}  # maps image_id to the coordinates of the detected box
    for im_id, result in zip(image_ids[0], results):
        vidstg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

    # create output dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4])):
        os.makedirs(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4]))
    # extract actual images from the video to process them adding boxes
    os.system(
        f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(args.output_dir, vid_path.split("/")[-1][:-4], "%05d.jpg")}'
    )
    for img_id in image_ids[0]:
        # load extracted image
        img_path = os.path.join(
            args.output_dir,
            vid_path.split("/")[-1][:-4],
            str(int(img_id) + 1).zfill(5) + ".jpg",
        )
        img = Image.open(img_path).convert("RGB")
        imgw, imgh = img.size
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(img, aspect="auto")

        if (
            pred_steds[0] <= img_id < pred_steds[1]
        ):  # add predicted box if the image_id is in the predicted start and end
            x1, y1, x2, y2 = vidstg_res[img_id]["boxes"][0]
            w = x2 - x1
            h = y2 - y1
            rect = plt.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False
            )
            ax.add_patch(rect)

        fig.set_dpi(100)
        fig.set_size_inches(imgw / 100, imgh / 100)
        fig.tight_layout(pad=0)

        # save image with eventual box
        fig.savefig(
            img_path,
            format="jpg",
        )
        plt.close(fig)

    # save video with tube
    os.system(
        f"ffmpeg -r {extracted_fps} -pattern_type glob -i '{os.path.join(args.output_dir, vid_path.split('/')[-1][:-4])}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(args.output_dir, vid_path.split('/')[-1])}"
    )
