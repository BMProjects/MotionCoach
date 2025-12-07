import numpy as np
from scipy.spatial import procrustes
from .base_analyzer import BaseSimilarityAnalyzer

class ProcrustesSimilarityAnalyzer(BaseSimilarityAnalyzer):
    
    def _normalize_pose(self, pose: np.ndarray):
        """对姿态进行平移和缩放归一化"""
        # 使用置信度>0.5的关键点进行归一化
        valid_points = pose[pose[:, 2] > 0.5][:, :2]
        if valid_points.shape[0] < 2:
            return None
        
        # 1. 平移到中心
        center = np.mean(valid_points, axis=0)
        centered_points = valid_points - center
        
        # 2. 尺度归一化 (Frobenius norm)
        norm = np.linalg.norm(centered_points, 'fro')
        if norm == 0:
            return centered_points
            
        normalized_points = centered_points / norm
        return normalized_points

    def compare(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """使用Procrustes分析比较两个姿态"""
        
        norm_pose1 = self._normalize_pose(pose1)
        norm_pose2 = self._normalize_pose(pose2)
        
        if norm_pose1 is None or norm_pose2 is None or norm_pose1.shape != norm_pose2.shape:
            # 如果有效点数量不一致或无法归一化，则认为不相似
            return 0.0

        try:
            # Procrustes分析返回 mtx1, mtx2, disparity
            # disparity(d) 是一个标准化的距离度量，范围在[0, 1]之间
            # d=0 表示完全相同，d=1表示完全不同（在归一化后）
            _, _, disparity = procrustes(norm_pose1, norm_pose2)
            
            # 将距离转换为相似度分数 [0, 100]
            similarity = (1 - disparity) * 100
            
        except (ValueError, TypeError):
             # Procrustes 可能会因矩阵问题失败
            similarity = 0.0
        
        return max(0.0, similarity)