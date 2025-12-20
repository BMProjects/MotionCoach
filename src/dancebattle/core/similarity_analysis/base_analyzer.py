from abc import ABC, abstractmethod
import numpy as np

class BaseSimilarityAnalyzer(ABC):
    """相似度分析器的抽象基类 (接口)"""

    @abstractmethod
    def compare(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        比较两个姿态的相似度。
        
        Args:
            pose1: 第一个姿态的关键点数组 (N, 3)。
            pose2: 第二个姿态的关键点数组 (N, 3)。
            
        Returns:
            一个0到100之间的相似度分数。
        """
        pass