from .kitti_y import KITTIDataset, KITTI_Raw
from .MiddEval import MiddEval
__datasets__ = {
    "kitti": KITTIDataset, 
    "kitti_raw": KITTI_Raw,
    "MiddEval": MiddEval
}
