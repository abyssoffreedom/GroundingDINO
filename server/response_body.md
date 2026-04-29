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
    "server_processing_ms": 180.7,
    "server_receive_ns": 330821982123456,
    "server_send_ns": 330821982304156
  }
}


当未开启可视化或无检测结果时，visualization_b64 字段不会出现。
results 顺序与请求中上传图片顺序一致。

说明：
1. boxes: 归一化坐标 [xmin, ymin, xmax, ymax]，范围 0-1，适用于客户端渲染。
2. boxes_coco: 像素坐标 [xmin, ymin, width, height]，适用于 COCO 指标评测。
3. 服务端聚合耗时使用 metrics.server_processing_ms，表示请求解码、预处理、推理、后处理和 JSON 打包的总时间。
4. server_receive_ns 和 server_send_ns 使用服务端 monotonic clock（Python time.perf_counter_ns），可与 UDP time-sync offset 校准后的客户端 monotonic 时间戳比较；它们不是系统显示时间或 Unix epoch 时间。
