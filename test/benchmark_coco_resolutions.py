import argparse
import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

class LetterBox:
    """
    Resizes image to a target square resolution using Letterboxing (preserve aspect ratio + pad).
    Standard Computer Vision approach (often used in YOLO).
    """
    def __init__(self, new_shape, color=(128, 128, 128)):
        self.new_shape = new_shape
        self.color = color

    def __call__(self, img, target=None):
        # img: PIL Image
        w, h = img.size
        if isinstance(self.new_shape, int):
            new_h, new_w = self.new_shape, self.new_shape
        else:
            new_h, new_w = self.new_shape

        # Scale ratio (new / old)
        r = min(new_w / w, new_h / h)
        
        # Compute padding
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))
        dw, dh = new_w - new_unpad_w, new_h - new_unpad_h

        # Center padding
        dw /= 2
        dh /= 2

        if (w, h) != (new_unpad_w, new_unpad_h):
            img = img.resize((new_unpad_w, new_unpad_h), Image.BICUBIC)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Pad image
        img = F.pad(img, (left, top, right, bottom), fill=self.color[0])
        
        # Store transformation info in target for coordinate restoration
        if target is not None:
            target['ratio'] = r
            target['pad'] = (left, top) # x, y
            target['letterbox_size'] = (new_h, new_w)
            
        return img, target

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target_new = self._transforms(img, target_new)

        return img, target_new

class PostProcessCocoGrounding(nn.Module):
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None):
        super().__init__()
        self.num_select = num_select
        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(tokenlizer(captions), tokenspanlist)
        
        # COCO 2017 category mapping
        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T
        
        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]
        
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

def load_model_func(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def evaluate_resolution(args, model, resolution, device):
    print(f"\n{'='*20}")
    print(f"Evaluating Resolution: {resolution}x{resolution}")
    print(f"{'='*20}")
    
    # 1. Define Transform with LetterBox
    transform = T.Compose([
        LetterBox(resolution),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 2. Setup DataLoader
    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    cfg = SLConfig.fromfile(args.config_file)
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(coco_api=dataset.coco, tokenlizer=tokenlizer)
    evaluator = CocoGroundingEvaluator(dataset.coco, iou_types=("bbox",), useCats=True)
    
    # 3. Construct Prompt (All 80 categories)
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print(f"Prompt constructed with {len(cat_list)} categories.")
    
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(device)
        bs = images.shape[0]
        input_captions = [caption] * bs
        
        with torch.no_grad():
            outputs = model(images, captions=input_captions)
        
        # Pass letterbox size to postprocessor to get pixels in padded image
        target_sizes = torch.stack([torch.tensor(t['letterbox_size']) for t in targets], dim=0).to(device)
        results = postprocessor(outputs, target_sizes)
        
        # 4. Restore Coordinates (Reverse Letterbox)
        for res, t in zip(results, targets):
            boxes = res['boxes']
            r = t['ratio']
            pad_w, pad_h = t['pad']
            orig_h, orig_w = t['orig_size']
            
            # Subtract padding
            boxes[:, [0, 2]] -= pad_w
            boxes[:, [1, 3]] -= pad_h
            # Unscale
            boxes /= r
            # Clip to original image dimensions
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_h)
            res['boxes'] = boxes

        cocogrounding_res = {target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)
        
        if (i+1) % 100 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(f"Processed {i+1}/{len(data_loader)} images. Time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.coco_eval["bbox"].stats.tolist()

def main():
    parser = argparse.ArgumentParser(description="Benchmark GroundingDINO on COCO with multiple resolutions")
    parser.add_argument("--config_file", "-c", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, default="weights/groundingdino_swint_ogc.pth", help="path to checkpoint file")
    parser.add_argument("--anno_path", type=str, required=True, help="coco annotation json")
    parser.add_argument("--image_dir", type=str, required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_model_func(args.config_file, args.checkpoint_path, args.device)
    model = model.to(args.device)
    
    resolutions = [800, 640, 512, 416, 256]
    all_results = {}
    
    for res in resolutions:
        stats = evaluate_resolution(args, model, res, args.device)
        all_results[res] = stats
        
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Resolution':<12} | {'AP':<6} | {'AP50':<6} | {'AP75':<6} | {'APs':<6} | {'APm':<6} | {'APl':<6}")
    print("-" * 80)
    for res in resolutions:
        s = all_results[res]
        # stats: 0:AP, 1:AP50, 2:AP75, 3:APs, 4:APm, 5:APl ...
        print(f"{res:<12} | {s[0]:.3f}  | {s[1]:.3f}  | {s[2]:.3f}  | {s[3]:.3f}  | {s[4]:.3f}  | {s[5]:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
