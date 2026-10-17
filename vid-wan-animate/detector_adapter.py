"""Permissively-licensed replacement for Wan2.2-Animate's bundled person detector.

wan/modules/animate/preprocess/pose2d.py builds its detector as:

    self.detector = Yolo(detector_checkpoint, device)   # Pose2d.__init__

`detector_checkpoint` points at `process_checkpoint/det/yolov10m.onnx` in the official weights
repo -- Ultralytics YOLOv10, AGPL-3.0 (confirmed against both github.com/THU-MIG/yolov10 and
github.com/ultralytics/ultralytics, 2026-10-17). We never download that file; see this worker's
README.md licence table.

Rather than edit the vendored pose2d.py, handler.py monkey-patches the class the function looks up
at call time:

    import pose2d                      # wan/modules/animate/preprocess/pose2d.py, on sys.path
    import detector_adapter
    pose2d.Yolo = detector_adapter.PersonDetector

`Pose2d.__init__`'s `Yolo(detector_checkpoint, device)` call resolves `Yolo` as a bare global in
`pose2d`'s own module namespace *every time it runs*, not at class-definition time -- so the patch
above is enough. Nothing in pose2d.py, pose2d_utils.py or process_pipepline.py changes.

External contract this class must satisfy (pose2d.py, class Pose2d.__call__, reproduced here
because it is the thing that must keep working):

    img, shape = detector.preprocess(image_rgb_uint8_HWC)   # one video frame at a time
    person = detector(img[None], shape[None])[0][0]         # batch of 1 -> image 0 -> best person
    bbox = person["bbox"]                                    # np.float64 [x1, y1, x2, y2, score]
                                                              # in ORIGINAL image pixel coordinates

`ViTPose.preprocess` (same file) then crops around that box; a low/negative score or a
too-small box makes it fall back to the full frame, so a conservative detector is safe.

Backend: torchvision's Faster R-CNN ResNet50 FPN v2, COCO-pretrained (BSD-3-Clause code, weights
hosted by the project at download.pytorch.org under the same licence -- see README.md). torchvision
is already a pinned base-image dependency, so this adds no new pip package. The weights are baked
into the image at build time (Dockerfile RUN step), so there is no first-request download and no
runtime dependency on download.pytorch.org being reachable.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

# torchvision's detection models keep the ORIGINAL 91-class COCO category ids (not the condensed
# 80-class reindexing YOLO/Ultralytics uses). "person" is category id 1 in both the official COCO
# `instances_*.json` files and torchvision's own label map -- no "+1" remap needed here, unlike
# Yolo.process_results's `bbox[-1] + 1 in cat_id` which exists only to undo YOLO's own reindexing.
_COCO_PERSON_LABEL = 1

_model_cache: dict[str, torch.nn.Module] = {}


def _get_model(device: str) -> torch.nn.Module:
    model = _model_cache.get(device)
    if model is None:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        model.eval().to(device)
        _model_cache[device] = model
    return model


class PersonDetector:
    """Drop-in replacement for `pose2d.Yolo`.

    Constructor signature matches `Yolo(checkpoint, device='cuda', **kwargs)` so
    `Pose2d.__init__`'s `Yolo(detector_checkpoint, device)` call needs no change anywhere.
    `checkpoint` is accepted and ignored: there is no file to load, the COCO weights are already
    resident in the image (see Dockerfile) and loaded once per process into `_model_cache`.
    """

    def __init__(self, checkpoint=None, device="cuda", threshold_conf: float = 0.05, **kwargs):
        if isinstance(device, torch.device):
            device = f"{device.type}:{device.index or 0}" if device.type == "cuda" else device.type
        self.device = device if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu"
        self.threshold_conf = threshold_conf
        self.model = _get_model(self.device)

    def preprocess(self, input_image):
        """Mirrors Yolo.preprocess's return shape: (image_data, [height, width]).

        No letterboxing: Faster R-CNN's own GeneralizedRCNNTransform resizes internally, so the
        full-resolution frame is passed through unmodified rather than forced into YOLO's fixed
        640x640 input (arguably better: no accuracy lost to a forced downscale on a 720p/1080p
        driving frame).
        """
        img = np.asarray(input_image)
        h, w = img.shape[:2]
        chw = np.ascontiguousarray(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return chw, np.array([h, w])

    def __call__(self, img_batch, shape_batch, **kwargs):
        return self.forward(img_batch, shape_batch, **kwargs)

    def forward(self, img_batch, shape_batch, **kwargs):
        """img_batch: (N, 3, H, W) float32 in [0, 1]. shape_batch: (N, 2) = [height, width].

        Returns a list of N per-image person lists, each a single-entry list (this adapter only
        ever reports the best person, which is all `Pose2d.__call__` reads: `...[0][0]["bbox"]`).
        """
        if isinstance(img_batch, torch.Tensor):
            img_batch = img_batch.cpu().numpy()
        if isinstance(shape_batch, torch.Tensor):
            shape_batch = shape_batch.cpu().numpy()

        # Full-frame fallback -- same shape and same "no detector" meaning (track_id -1, score -1)
        # as Yolo.forward's own default, which ViTPose.preprocess treats as "use the whole image".
        results = [
            [{"bbox": np.array([0.0, 0.0, float(shape_batch[i][1]), float(shape_batch[i][0]), -1.0]),
              "track_id": -1}]
            for i in range(len(img_batch))
        ]

        with torch.no_grad():
            tensors = [torch.from_numpy(im).to(self.device) for im in img_batch]
            predictions = self.model(tensors)

        for i, pred in enumerate(predictions):
            boxes = pred["boxes"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            keep = (labels == _COCO_PERSON_LABEL) & (scores >= self.threshold_conf)
            boxes, scores = boxes[keep], scores[keep]
            if len(boxes) == 0:
                continue
            # select_type="max": Pose2d's default Yolo(...) call takes every other knob at its
            # default too, and Yolo's default select_type is "max" -- the largest-area person.
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            x1, y1, x2, y2 = (float(v) for v in boxes[best])
            results[i] = [{"bbox": np.array([x1, y1, x2, y2, float(scores[best])]), "track_id": 0}]

        return results
