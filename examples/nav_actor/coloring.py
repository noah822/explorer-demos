# static mapping from 0-indexed class label to color in RGB format
import matplotlib.cm as cm
import numpy as np
color_array = cm.rainbow(np.linspace(0, 1, 16))

COCO_COLORING = [
    (255 * c[:3]).astype(np.uint8) for c in color_array
]