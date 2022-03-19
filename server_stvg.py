#!/usr/bin/env python
import os
import json
import torch
import random
import cherrypy
import numpy as np
import ffmpeg
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from main import get_args_parser
from util.misc import NestedTensor
from datasets.video_transforms import make_video_transforms, prepare
from models.postprocessors import PostProcessSTVG, PostProcess
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.tubedetr import TubeDETR

DATA_PATH = "/srv/"
CODE_PATH = os.path.abspath(os.path.dirname(__file__))


class Server(object):
    def __init__(
        self,
        model,
        model_ckpt,
        annotations,
        max_videos,
        fps=5,
        video_max_len=200,
        resolution=224,
        stride=2,
    ):
        """
        :param model: model used for the demo
        :param model_ckpt: path to weights for the model
        :param annotations: test set annotations for caption / start / end placeholders
        :param max_videos: maximum number of videos in the demo
        :param fps: number of frames per second
        :param video_max_len: maximum number of frames to be extracted from a video
        :param resolution: spatial frame resolution
        :param stride: pool size k
        """

        # load weights for the first model on CPU
        self.model = model
        checkpoint = torch.load(model_ckpt, map_location="cpu")
        if "model_ema" in checkpoint:
            checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"][
                "query_embed.weight"
            ][:1]
            if "transformer.time_embed.te" in checkpoint["model_ema"]:
                del checkpoint["model_ema"]["transformer.time_embed.te"]
            model.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            checkpoint["model"]["query_embed.weight"] = checkpoint["model"][
                "query_embed.weight"
            ][:1]
            if "transformer.time_embed.te" in checkpoint["model"]:
                del checkpoint["model"]["transformer.time_embed.te"]
            model.load_state_dict(checkpoint["model"], strict=False)
        print("checkpoint loaded")
        self.model.eval()

        self.annotations = annotations
        self.vid2idx = (
            {}
        )  # map original_video_id to a list of default annotations corresponding to this video
        for i, x in enumerate(annotations["videos"]):
            self.vid2idx[x["original_video_id"]] = self.vid2idx.get(
                x["original_video_id"], []
            ) + [i]
        self.all_video_ids = list(
            dict.fromkeys([x["original_video_id"] for x in self.annotations["videos"]])
        )[
            :max_videos
        ]  # list of original video ids to show
        self.max_videos = max_videos
        self.transforms = make_video_transforms(
            "test", cautious=True, resolution=resolution
        )
        self.video_max_len = video_max_len
        self.stride = stride
        self.postprocessors = {"vidstg": PostProcessSTVG(), "bbox": PostProcess()}
        self.fps = fps

    @cherrypy.expose
    def index(self):
        index_html = '<head><link rel="icon" href="https://antoyang.github.io/img/favicon.ico" type="image/x-icon"/>'
        index_html += '<link href="https://antoyang.github.io/css/bootstrap.min.css" rel="stylesheet"></head>'
        index_html += "<center><h1> <a href='https://antoyang.github.io/tubedetr.html'> TubeDETR </a> Spatio-Temporal Video Grounding Demo </h1></center>"
        index_html += "<center><h2> Choose a video for which you want to localize a query </h2></center>"
        index_html += "<center><h3> Default queries are from the VidSTG test set annotations. Nothing is pre-computed for these videos. </h3></center><br>"
        index_html += '<div class="container">'  # grid of videos
        for i, vid in enumerate(self.all_video_ids):
            thumbnail_path = f"http://stvg.paris.inria.fr/data/image/{vid}.jpg"
            if i % 4 == 0:  # 4 videos per row
                index_html += '<div class="row">'
            index_html += '<div class="col-md-3 col-sm-12"><center><a href="stvg?video_id={}"><img src={} height="180" width="240"></img></a><br>'.format(
                vid, thumbnail_path
            )
            index_html += '<a href="stvg?video_id={}">{}</a></center></div>'.format(
                vid, vid
            )
            if (i % 4 == 3) or (
                i == min(len(self.all_video_ids), self.max_videos) - 1
            ):  # end of row
                index_html += "</div><br><br>"
        index_html += "</div>"

        index_html += "<center><a href='reload' class='btn btn-primary btn-lg active'>More videos!</a></center><br>"
        index_html += "<center><h2> Built by <a href='https://antoyang.github.io/'> Antoine Yang </a> </h2> </center><br>"
        return index_html

    @cherrypy.expose
    def stvg(self, video_id, start=0, end=30, question=""):
        if video_id not in self.all_video_ids:
            return (
                f'Video {video_id} is not available, <a href="/">go back to index</a>.'
            )
        html_path = os.path.join(CODE_PATH, "server_stvg.html")
        with open(html_path, "r") as f:
            html = f.read()
        if not question:  # put default data for question, start and end
            flag = False
            idx = np.random.randint(len(self.vid2idx[video_id]))
            video_data = self.annotations["videos"][self.vid2idx[video_id][idx]]
            question = video_data["caption"]
            start_frame = video_data["start_frame"]
            end_frame = video_data["end_frame"]
            video_fps = video_data["fps"]
            start = start_frame / video_fps
            end = end_frame / video_fps
        else:
            flag = True  # a question is asked
            video_data = self.annotations["videos"][
                self.vid2idx[video_id][0]
            ]  # just to load metadata
            start = float(start)
            end = float(end)
        vid_path = f"http://stvg.paris.inria.fr/data/video/{video_data['video_path']}"
        html = html.format(vid_path, start, end, video_id, start, end, question)
        if flag:
            # video extraction
            ss = start
            t = end - start
            extracted_fps = min((self.fps * t), self.video_max_len) / t
            cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=extracted_fps)
            out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                capture_stdout=True, quiet=True
            )
            w = video_data["width"]
            h = video_data["height"]
            images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
            image_ids = [[k for k in range(len(images_list))]]

            # video transforms
            empty_anns = []  # empty targets as placeholders for the transforms
            placeholder_target = prepare(w, h, empty_anns)
            placeholder_targets_list = [placeholder_target] * len(images_list)
            images, targets = self.transforms(images_list, placeholder_targets_list)

            samples = NestedTensor.from_tensor_list([images], False)
            if self.stride:
                samples_fast = samples
                samples = NestedTensor.from_tensor_list(
                    [images[:, :: self.stride]], False
                )
            else:
                samples_fast = None
            durations = [len(targets)]
            captions = [question]

            with torch.no_grad():  # forward
                # encoder
                memory_cache = self.model(
                    samples,
                    durations,
                    captions,
                    encode_and_save=True,
                    samples_fast=samples_fast,
                )
                # decoder
                outputs = self.model(
                    samples,
                    durations,
                    captions,
                    encode_and_save=False,
                    memory_cache=memory_cache,
                )

                pred_steds = self.postprocessors["vidstg"](
                    outputs, image_ids, video_ids=[0]
                )[
                    0
                ]  # (start, end) in terms of image_ids
                orig_target_sizes = torch.stack(
                    [t["orig_size"] for t in targets], dim=0
                )
                results = self.postprocessors["bbox"](outputs, orig_target_sizes)
                vidstg_res = {}  # maps image_id to the coordinates of the detected box
                for im_id, result in zip(image_ids[0], results):
                    vidstg_res[im_id] = {
                        "boxes": [result["boxes"].detach().cpu().tolist()]
                    }

            # create output dir
            vid_dir = os.path.join(DATA_PATH, "demos", str(video_id))
            if os.path.exists(vid_dir):
                shutil.rmtree(vid_dir)
            os.makedirs(vid_dir)

            # extract actual images from the video to process them adding boxes
            os.system(
                f"ffmpeg -y -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {vid_dir}/%05d.jpg"
            )

            for img_id in image_ids[0]:
                # load extracted image
                img_path = os.path.join(
                    vid_dir,
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
                f"ffmpeg -y -r {extracted_fps} -pattern_type glob -i '{vid_dir}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(DATA_PATH, 'demos', str(video_id) + '.mp4')}"
            )

            # plot generated videos with output tube
            html += '<div class="col-sm-offset-2 col-sm-8"> <b> Question input </b>: {} <br> <b> Start input </b>: {} <br> <b> End input </b>: {} <br> <b> Predicted Spatio-Temporal Tube </b>: <br> </div>'.format(
                question, start, end
            )
            html += '<div class="row"><div class="col-sm-offset-2 col-sm-6"><video width="100%" height="360" controls="controls" preload="metadata"><source  src="{}#t={},{}" type="video/mp4"></video></div></div>'.format(
                f"http://stvg.paris.inria.fr/data/demos/{video_id}.mp4",
                start,
                end,
            )

        return html + "</div><br><br></body></html>"

    @cherrypy.expose
    def reload(self):  # same as index after a randomizing the videos
        self.all_video_ids = random.sample(list(self.all_video_ids), self.max_videos)

        index_html = '<head><link rel="icon" href="https://antoyang.github.io/img/favicon.ico" type="image/x-icon"/>'
        index_html += '<link href="https://antoyang.github.io/css/bootstrap.min.css" rel="stylesheet"></head>'
        index_html += "<center><h1> <a href='https://antoyang.github.io/tubedetr.html'> TubeDETR </a> Spatio-Temporal Video Grounding Demo </h1></center>"
        index_html += "<center><h2> Choose a video for which you want to localize a query </h2></center>"
        index_html += "<center><h3> Default queries are from the VidSTG test set annotations. Nothing is pre-computed for these videos. </h3></center><br>"
        index_html += '<div class="container">'  # grid of videos

        for i, vid in enumerate(self.all_video_ids):
            thumbnail_path = f"http://stvg.paris.inria.fr/data/image/{vid}.jpg"
            if i % 4 == 0:  # 4 videos per row
                index_html += '<div class="row">'
            index_html += '<div class="col-md-3 col-sm-12"><center><a href="stvg?video_id={}"><img src={} height="180" width="240"></img></a><br>'.format(
                vid, thumbnail_path
            )
            index_html += '<a href="stvg?video_id={}">{}</a></center></div>'.format(
                vid, vid
            )
            if (i % 4 == 3) or (
                i == min(len(self.all_video_ids), self.max_videos) - 1
            ):  # end of row
                index_html += "</div><br><br>"
        index_html += "</div>"

        index_html += "<center><a href='reload' class='btn btn-primary btn-lg active'>More videos!</a></center><br>"
        index_html += "<center><h2> Built by <a href='https://antoyang.github.io/'> Antoine Yang </a> </h2> </center><br>"
        return index_html


def run():
    parser = argparse.ArgumentParser(
        "TubeDETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    port = args.port
    cherrypy.config.update({"server.socket_port": port})
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    conf = {"/data": {"tools.staticdir.on": True, "tools.staticdir.dir": DATA_PATH}}

    annotations = json.load(open(os.path.join(DATA_PATH, "test.json"), "r"))

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

    print(f"http server is running at port {port}")
    cherrypy.quickstart(
        Server(
            model,
            args.load,
            annotations,
            args.batch_size,
            args.fps,
            args.video_max_len,
            args.resolution,
            args.stride,
        ),
        "/",
        conf,
    )


if __name__ == "__main__":
    run()
