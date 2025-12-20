from abc import ABC, abstractmethod
import numpy as np

class BasePoseEstimator(ABC):
    """姿态估计器的抽象基类 (接口)"""
    
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """
        对单帧图像进行姿态估计。
        
        Args:
            frame: BGR格式的图像帧 (np.ndarray)。
            
        Returns:
            一个numpy数组，形状为 (N, 3)，代表N个关键点的(x, y, visibility)，
            如果未检测到姿态，则返回None。
        """
        pass