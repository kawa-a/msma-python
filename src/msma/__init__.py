from ._api import msma
from .s4pca import S4PCA

__all__ = ["msma", "S4PCA"]
__version__ = "0.1.0rc1"
from .model_selection import MSMASelectionResult, select_msma

__all__ += ["select_msma", "MSMASelectionResult"]
