import cv2
import numpy as np
import os
'''
1. 视频分帧与光流计算
输入：声纳视频（连续帧序列）。

输出：

空间流输入：单帧图像（RGB或灰度）。

时间流输入：光流图（Optical Flow）或帧间差分图（Frame Difference）。

'''
def compute_optical_flow(prev_frame, next_frame):
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    # 计算密集光流（Farneback方法）
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 将光流转换为RGB可视化（可选）
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_frame)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

# 计算相邻帧的光流
output_dir = "frames"
frame_files = sorted(os.listdir(output_dir))
# 创建保存光流的文件夹
flow_folder = "flow"
os.makedirs(flow_folder, exist_ok=True)

for i in range(1, len(frame_files)):
    prev_frame = cv2.imread(os.path.join(output_dir, frame_files[i-1]))
    next_frame = cv2.imread(os.path.join(output_dir, frame_files[i]))
    flow = compute_optical_flow(prev_frame, next_frame)
     # 保存光流到flow文件夹
    flow_path = os.path.join(flow_folder, f"flow_{i-1:04d}.png")
    cv2.imwrite(flow_path, flow)
print(f"光流计算完成，结果保存在 {flow_folder} 文件夹中")