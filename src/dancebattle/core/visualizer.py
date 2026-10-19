import cv2
import numpy as np
from tqdm import tqdm

# MediaPipe 33 点姿态的标准连接关系（mediapipe>=1.0 已移除 mp.solutions，此处直接列出同一组连接）
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
])

def draw_pose(image: np.ndarray, landmarks: np.ndarray | None):
    """
    在图像上绘制姿态关键点和骨骼。
    
    Args:
        image: 要绘制的图像帧 (BGR格式)。
        landmarks: 姿态关键点数组 (N, 3)，格式为 (x, y, visibility)。
                   x, y 是 0-1 范围内的归一化坐标。
    """
    if landmarks is None:
        return

    h, w, _ = image.shape
    
    # 绘制连接线
    if POSE_CONNECTIONS:
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            # 检查关键点是否有效（在landmarks数组范围内且置信度高）
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                if start_landmark[2] > 0.5 and end_landmark[2] > 0.5: # 检查visibility
                    start_point = (int(start_landmark[0] * w), int(start_landmark[1] * h))
                    end_point = (int(end_landmark[0] * w), int(end_landmark[1] * h))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # 绘制关键点
    for landmark in landmarks:
        if landmark[2] > 0.5: # 检查visibility
            center = (int(landmark[0] * w), int(landmark[1] * h))
            cv2.circle(image, center, 5, (0, 0, 255), -1) # -1表示实心圆


def create_comparison_video(
    video_path_a: str,
    video_path_b: str,
    poses_a: list,
    poses_b: list,
    scores: list,
    frame_offset: int,
    output_path: str,
    fps: float,
    target_height: int = 720
):
    """
    创建并排的姿态比较视频。

    Args:
        video_path_a: 视频A的路径。
        video_path_b: 视频B的路径。
        poses_a: 视频A的姿态序列。
        poses_b: 视频B的姿态序列。
        scores: 每帧的相似度分数列表。
        frame_offset: 视频B相对于A的帧偏移量。
        output_path: 输出视频的保存路径。
        fps: 输出视频的帧率。
        target_height: 输出视频中单个视频画面的目标高度。
    """
    print(f"Starting to create comparison video at {output_path}...")

    cap_a = cv2.VideoCapture(video_path_a)
    cap_b = cv2.VideoCapture(video_path_b)

    # # 确定对齐后的起始帧
    # start_a, start_b = 0, 0
    # if frame_offset > 0:
    #     start_b = frame_offset
    # else:
    #     start_a = -frame_offset
    
    # 确定对齐后的起始帧
    start_a, start_b = 0, 0
    if frame_offset > 0: # B 晚于 A，需要跳过A视频前面的帧
        start_a = frame_offset
    else: # A 晚于 B，需要跳过B视频前面的帧
        start_b = -frame_offset

    # 将视频指针移动到起始帧
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_a)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_b)

    # 初始化视频写入器
    # 我们需要先读取一帧来确定最终的宽度
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    if not ret_a or not ret_b:
        print("Error: Could not read initial frames from videos.")
        return

    def resize_to_height(image, height):
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(height * aspect_ratio)
        return cv2.resize(image, (new_width, height))

    frame_a_resized = resize_to_height(frame_a, target_height)
    frame_b_resized = resize_to_height(frame_b, target_height)
    
    combined_width = frame_a_resized.shape[1] + frame_b_resized.shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, target_height))

    # 重置视频指针以开始循环
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_a)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_b)
    
    num_frames_to_compare = len(scores)
    for i in tqdm(range(num_frames_to_compare), desc="Generating comparison video"):
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a or not ret_b:
            break

        # 绘制姿态
        draw_pose(frame_a, poses_a[start_a + i])
        draw_pose(frame_b, poses_b[start_b + i])

        # 调整大小并拼接
        frame_a_resized = resize_to_height(frame_a, target_height)
        frame_b_resized = resize_to_height(frame_b, target_height)
        combined_frame = np.hstack((frame_a_resized, frame_b_resized))

        # 添加相似度分数文本
        score = scores[i]
        score_text = f"Similarity: {score:.2f}%"
        
        # 计算文本位置
        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        text_x = (combined_frame.shape[1] - text_size[0]) // 2
        text_y = combined_frame.shape[0] - 30 # 底部

        # 添加黑色背景以提高可读性
        cv2.rectangle(combined_frame, (text_x - 10, text_y - text_size[1] - 10), 
                      (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(combined_frame, score_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        video_writer.write(combined_frame)

    # 释放资源
    cap_a.release()
    cap_b.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("Comparison video saved successfully.")