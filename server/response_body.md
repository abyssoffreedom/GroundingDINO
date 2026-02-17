{
  "request_id": "req-001",
  "device": "cuda",
  "results": [
    {
      "image_id": "img001",
      "boxes": [
        [0.1205, 0.0882, 0.3601, 0.4207],
        [0.512, 0.064, 0.7003, 0.3009]
      ],
      "boxes_coco": [
        [120.5, 88.2, 239.6, 332.5],
        [512.0, 64.0, 188.3, 236.9]
      ],
      "phrases": ["person", "dog"],
      "scores": [0.91, 0.82],
      "visualization_b64": "optional-base64-when-return_visualization-true"
    },
    {
      "image_id": "img002",
      "boxes": [],
      "boxes_coco": [],
      "phrases": [],
      "scores": []
    }
  ],
  "metrics": {
    "forward_ms": 123.4,
    "server_e2e_ms": 180.7,
    "t_server_request_received": 1710000000000.0,
    "t_server_response_done": 1710000000180.7
  }
}


当未开启可视化或无检测结果时，visualization_b64 字段不会出现。
results 顺序与请求中上传图片顺序一致。

说明：
1. boxes: 归一化坐标 [xmin, ymin, xmax, ymax]，范围 0-1，适用于客户端渲染。
2. boxes_coco: 像素坐标 [xmin, ymin, width, height]，适用于 COCO 指标评测。
3. 顶层 forward_ms 已移除，请使用 metrics.forward_ms。
