from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt') 
results = model('./vkitti_rgb/fog/frames/rgb/Camera_0/rgb_00225.jpg')
print(results)
results[0].show()
