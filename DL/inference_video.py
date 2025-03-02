import torch
import cv2
import time
import numpy as np
from models.dual_stream_net import DualStreamNet
from utils.data_utils import generate_pseudo_label, preprocess_video_frames
from utils.metrics import calculate_miou, calculate_fps
from config import Config

def inference_video(config, video_source=None):
    # 初始化模型
    model = DualStreamNet().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()

    # 打开视频源（文件或摄像头）
    if video_source is None:
        cap = cv2.VideoCapture(config.VIDEO_PATH)  # 使用配置文件中的视频文件
    else:
        cap = cv2.VideoCapture(video_source)  # 0 表示默认摄像头，或其他视频文件路径

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # 帧缓冲区
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        # 转换为灰度图（假设声呐视频为单通道）
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_buffer.append(frame_gray)
        
        if len(frames_buffer) == 3:
            # 预处理帧
            static_input, dynamic_input = preprocess_video_frames(frames_buffer, config.IMG_SIZE)
            static_input = static_input.to(config.DEVICE)
            dynamic_input = dynamic_input.to(config.DEVICE)
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                pred = model(static_input, dynamic_input)
            inference_time = time.time() - start_time
            
            # 后处理
            pred_np = pred.cpu().numpy()[0, 0] * 255  # 转换为显示格式
            pred_display = pred_np.astype(np.uint8)
            
            # 计算伪标签和指标
            pseudo_label = generate_pseudo_label(frames_buffer)
            pseudo_label = cv2.resize(pseudo_label, config.IMG_SIZE) / 255.0
            miou = calculate_miou(pred_np / 255.0, pseudo_label)
            fps = calculate_fps(inference_time, 1)
            
            # 显示结果
            cv2.imshow('Original Frame', frames_buffer[1])  # 显示中间帧
            cv2.imshow('Prediction', pred_display)
            print(f'mIoU: {miou:.4f}, FPS: {fps:.2f}')
            
            # 移除最早的帧，保持缓冲区大小为3
            frames_buffer.pop(0)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = Config()
    inference_video(config)  # 默认使用 config.VIDEO_PATH
    # 若要使用摄像头，调用：inference_video(config, video_source=0)