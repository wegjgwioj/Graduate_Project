import cv2
import numpy as np
# 读取空间流视频
spatial_video = cv2.VideoCapture("spatial_video.mp4")

# 获取视频的帧率、宽度和高度
fps = spatial_video.get(cv2.CAP_PROP_FPS)
width = int(spatial_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(spatial_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个VideoWriter对象来写入时间流视频
# 修改编解码器为XVID
fourcc = cv2.VideoWriter_fourcc(*'XVID')
temporal_video = cv2.VideoWriter("temporal_video.avi", fourcc, fps, (width, height), isColor=False)

# 读取第一帧
ret, prev_frame = spatial_video.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
print(prev_gray.shape)

while True:
    # 读取下一帧
    ret, frame = spatial_video.read()
    if not ret:
        break
    
    # 将当前帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("读出的灰度图像：",gray.shape)
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 可视化光流
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_image = np.zeros_like(prev_frame)
    flow_image[..., 0] = angle * 180 / np.pi / 2
    flow_image[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_HSV2BGR)
    
    # 将光流图像转换为灰度图像
    flow_gray = cv2.cvtColor(flow_image, cv2.COLOR_BGR2GRAY)
    
    # 写入时间流视频
    temporal_video.write(flow_gray)
    print("写入的灰度图像：",flow_gray.shape)
    # 更新前一帧
    prev_gray = gray

# 释放资源
spatial_video.release()
temporal_video.release()
cv2.destroyAllWindows()