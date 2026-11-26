from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision  # noqa: F401  # Match the M6 box suppression backend.
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from fractureguard.evaluation.m5 import _load_policy

from .imaging import VIEWS, normalized
from .security import digest


class FrozenPipeline:
    def __init__(self, model_path, policy_path, model_sha, policy_sha, preprocessing):
        if (
            digest(model_path.read_bytes()) != model_sha
            or digest(policy_path.read_bytes()) != policy_sha
        ):
            raise ValueError("The installed model and policy must match the approved artifacts")
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.policy, self.calibrator = _load_policy(policy_path)
        self.letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)
        self.versions = {
            "model_sha256": model_sha,
            "policy_sha256": policy_sha,
            "preprocessing": preprocessing,
            "runtime": ort.__version__,
        }

    def predict_view(self, png):
        with Image.open(BytesIO(png), formats=["PNG"]) as image:
            pixels = np.asarray(image)
        normalized_image = cv2.cvtColor(normalized(pixels), cv2.COLOR_GRAY2RGB)
        fitted = self.letterbox(image=normalized_image)
        tensor = np.ascontiguousarray(fitted.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.shape[:2] != (1, 5) or not np.isfinite(output).all():
            raise ValueError("Unexpected model output")
        boxes = non_max_suppression(torch.from_numpy(output), 0.05, 0.5)[0]
        scale_boxes(tensor.shape[-2:], boxes[:, :4], pixels.shape)
        retained = [row for row in boxes.tolist() if row[0] < row[2] and row[1] < row[3]]
        return retained, min(pixels.shape) < 256

    def predict(self, pair):
        if set(pair) != set(VIEWS):
            raise ValueError("Exactly two verified views are required")
        boxes, scores, warnings = [], {}, []
        for view, label in (("frontal", "AP/PA"), ("lateral", "Lateral")):
            detections, low_detail = self.predict_view(pair[view])
            scores[view] = max((row[4] for row in detections), default=0.0)
            for index, row in enumerate(detections):
                boxes.append(
                    {
                        "id": f"model-{view}-{index}",
                        "view": view,
                        "coords": row[:4],
                        "status": "preliminary",
                        "score": row[4],
                    }
                )
            if low_detail:
                warnings.append(f"The {label} image has limited pixel resolution.")
        score = max(scores.values())
        probability = float(self.calibrator.predict([score])[0])
        if (
            warnings
            or self.policy["negative_threshold"] <= probability < self.policy["positive_threshold"]
        ):
            outcome, label = "uncertain_manual_review", "Uncertain"
        elif probability >= self.policy["positive_threshold"]:
            outcome, label = "suspected_fracture", "Suspected fracture"
        else:
            outcome, label = "no_fracture_detected", "No fracture detected by model"
        return {
            "boxes": boxes,
            "view_scores": scores,
            "fused_max_score": score,
            "probability": probability,
            "score_label": f"{probability:.1%}",
            "outcome": outcome,
            "label": label,
            "warnings": warnings,
            **self.versions,
        }
