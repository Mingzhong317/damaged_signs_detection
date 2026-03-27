# -*- coding: utf-8 -*-
"""
det_cls.py
Step-by-step pipeline:
1) YOLOv5 object detection
2) Crop → resize 224×224
3) YOLOv8 classification (Damaged / Normal)
4) Dual-threshold post-filter, draw boxes & save JSON
"""

import sys, json, time, argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------- 1 解析參數 / CLI argparser ----------
def parse_opt():
    parser = argparse.ArgumentParser(
        description="YOLOv5 detection + YOLOv8 classification integrator")
    parser.add_argument("--det-weights", type=str,
                        default="yolov5s.pt", help="detection .pt")
    parser.add_argument("--cls-weights", type=str,
                        default="yolov8s-cls.pt", help="classification .pt")
    parser.add_argument("--img-dir", type=str,
                        default="../data/test/images", help="input image folder")
    parser.add_argument("--out-dir", type=str,
                        default="outputs/infer_out", help="output images folder")
    parser.add_argument("--json-path", type=str,
                        default="outputs/det_cls_results.json", help="output JSON")
    parser.add_argument("--conf-det", type=float, default=0.25,
                        help="detection confidence threshold")
    parser.add_argument("--conf-cls", type=float, default=0.6,
                        help="classification confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=640,
                        help="inference size for detection")
    parser.add_argument("--device", type=str, default="",
                        help="GPU id (0,1,…) or 'cpu'")
    return parser.parse_args()

# ---------- 2 載模型 / Load models ----------
def load_models(det_w, cls_w, device):
    # 加入 yolov5 repo 到 PYTHONPATH
    ROOT = Path(__file__).resolve().parents[1]
    YOLOV5_DIR = '../yolov5'
    sys.path.append(str(YOLOV5_DIR))

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    device = select_device(device)

    det_model = DetectMultiBackend(det_w, device=device, dnn=False)
    det_model.eval()
    raw_stride = det_model.stride                 # 可能是 int / list / tensor
    if hasattr(raw_stride, "max"):                # list / tensor → 取最大值
        stride = int(raw_stride.max())
    else:                                         # 已經是 int
        stride = int(raw_stride)

    # YOLOv8-cls (Ultralytics)
    from ultralytics import YOLO
    cls_model = YOLO(str(cls_w))      # ultralytics 將自動偵測 CPU / GPU
    cls_model.fuse()

    return det_model, cls_model, stride, device

# ---------- 3 偵測 → 回傳 NMS 後結果 ----------
def detect(det_model, img0, stride, img_size, conf_det, iou_thres, device):
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    img = letterbox(img0, new_shape=img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)       # BGR → RGB / HWC → CHW
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = det_model(im)

    pred = non_max_suppression(pred, conf_det, iou_thres)[0]

    if pred is None or len(pred) == 0:
        return []

    pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img0.shape)
    return pred.cpu().numpy()

# ---------- 4 主流程 ----------
def main():
    opt = parse_opt()
    det_w = Path(opt.det_weights)
    cls_w = Path(opt.cls_weights)

    # 準備路徑
    img_dir  = Path(opt.img_dir)
    out_dir  = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(opt.json_path)

    # 載入模型
    det_model, cls_model, stride, device = load_models(det_w, cls_w,
                                                       opt.device)
    det_names = det_model.names             # detection class list
    cls_names = cls_model.names             # ['Damaged', 'Normal'] (assumed)

    t0 = time.time()
    json_all = []

    # 逐張影像
    for img_path in sorted(img_dir.glob('*.*')):
        img0 = cv2.imread(str(img_path))    # BGR
        if img0 is None:
            print(f"⚠️  Cannot read {img_path}")
            continue

        # --- 4-1 Object Detection ---
        preds = detect(det_model, img0, stride, opt.img_size,
                       opt.conf_det, opt.iou_thres, device)

        detections = []

        # --- 4-2 Classification per bbox ---
        for *xyxy, conf_det, cls_det_id in preds:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img0[y1:y2, x1:x2]          # BGR
            if crop.size == 0:                 # invalid crop
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_res = cv2.resize(crop_rgb, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
            # ultralytics 支援 ndarray
            cls_res = cls_model(crop_res, verbose=False)[0]
            cls_probs = cls_res.probs.data.cpu().numpy()
            cls_id = int(cls_probs.argmax())
            conf_cls = float(cls_probs[cls_id])
            cls_label = cls_names[cls_id]

            # --- 4-3 Dual-threshold decision ---
            if cls_label.lower() == 'damaged' and conf_cls > opt.conf_cls:
                final_label = 'Damaged'
                color = (0, 0, 255)            # Red
            else:
                final_label = 'Normal'
                color = (0, 255, 0)            # Green

            # --- 4-4 Draw ---
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            txt = (f"{det_names[int(cls_det_id)]}|{final_label} "
                   f"{conf_cls:.2f}")
            cv2.putText(img0, txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- 4-5 Collect JSON entry ---
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "cls_det_id": int(cls_det_id),
                "cls_det_name": det_names[int(cls_det_id)],
                "det_conf": round(float(conf_det), 4),
                "cls_id": cls_id,
                "cls_label": cls_label,
                "cls_conf": round(conf_cls, 4),
                "final_label": final_label
            })

        # 儲存圖片
        out_img_path = out_dir / img_path.name
        cv2.imwrite(str(out_img_path), img0)

        # 加進總 JSON
        json_all.append({
            "image": img_path.name,
            "detections": detections
        })

    # ---------- 5 輸出 JSON ----------
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_all, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    fps = len(json_all) / elapsed if elapsed else 0
    print(f"✅ Finished {len(json_all)} images in {elapsed:.2f}s "
          f"({fps:.2f} FPS_e2e)")
    print(f"📄 Results JSON: {json_path}")
    print(f"🖼  Annotated images in: {out_dir}")

if __name__ == "__main__":
    main()
