curl -X POST http://localhost:8000/v1/detect \
  -F request_id=req-001 \
  -F use_gpu=true \
  -F prompt_text="person . dog ." \
  -F prompt_token_spans='[[[0,6]],[[9,12]]]' \
  -F box_threshold=0.35 \
  -F text_threshold=0.25 \
  -F nms_threshold=0.5 \
  -F max_detections=100 \
  -F return_visualization=false \
  -F image_ids='["img001","img002"]' \
  -F images=@/path/to/photo1.jpg \
  -F images=@/path/to/photo2.jpg




prompt_token_spans、image_ids 可省略；若省略 image_ids，服务端用上传文件名当作 image_id。