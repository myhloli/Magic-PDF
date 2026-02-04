# Copyright (c) Opendatalab. All rights reserved.
import torch
import numpy as np
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
)


class PPDocLayoutV3Model:
    def __init__(
