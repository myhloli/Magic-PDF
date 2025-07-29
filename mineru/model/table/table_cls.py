from pathlib import Path

import cv2
import numpy as np
import onnxruntime


class TableClsModel:
    def __init__(self):
        self.sess = onnxruntime.InferenceSession(
            Path.joinpath(Path(__file__).parent, "onnx", "table_cls.onnx")
        )
        self.less_length = 256
        self.cw, self.ch = 224, 224
        self.std = [0.229, 0.224, 0.225]
        self.scale = 0.00392156862745098
        self.mean = [0.485, 0.456, 0.406]
        self.labels = ["wired_table", "wireless_table"]

    def preprocess(self, img):
        # PIL图像转cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 放大图片，使其最短边长为256
        h, w = img.shape[:2]
        scale = 256 / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        img = cv2.resize(img, (w_resize, h_resize), interpolation=1)
        # 调整为224*224的正方形
        h, w = img.shape[:2]
        cw, ch = 224, 224
        x1 = max(0, (w - cw) // 2)
        y1 = max(0, (h - ch) // 2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        if w < cw or h < ch:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = img[y1:y2, x1:x2, ...]
        # 正则化
        split_im = list(cv2.split(img))
        std = [0.229, 0.224, 0.225]
        scale = 0.00392156862745098
        mean = [0.485, 0.456, 0.406]
        alpha = [scale / std[i] for i in range(len(std))]
        beta = [-mean[i] / std[i] for i in range(len(std))]
        for c in range(img.shape[2]):
            split_im[c] = split_im[c].astype(np.float32)
            split_im[c] *= alpha[c]
            split_im[c] += beta[c]
        img = cv2.merge(split_im)
        # 5. 转换为 CHW 格式
        img = img.transpose((2, 0, 1))
        imgs = [img]
        x = np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x

    def predict(self, img):
        x = self.preprocess(img)
        (result,) = self.sess.run(None, {"x": x})
        label = self.labels[np.argmax(result)]
        return label
