import numpy as np
from image_preprocessing_module import image_preprocessing

# Add this compilation in systemd service
image_preprocessing(np.ones((128, 128), dtype=np.float32))
