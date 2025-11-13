{
  "request_id": "req-001",
  "device": "cuda",
  "results": [
    {
      "image_id": "img001",
      "boxes": [
        [120.5, 88.2, 360.1, 420.7],
        [512.0, 64.0, 700.3, 300.9]
      ],
      "phrases": ["person", "dog"],
      "scores": [0.91, 0.82],
      "visualization_b64": "optional-base64-when-return_visualization-true"
    },
    {
      "image_id": "img002",
      "boxes": [],
      "phrases": [],
      "scores": []
    }
  ]
}


当未开启可视化或无检测结果时，visualization_b64 字段不会出现。
results 顺序与请求中上传图片顺序一致。