from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
from typing import Tuple, List
from tools import Image

Coord = Tuple[int, int]
TopLeft = Coord
BottomRight = Coord

Marker = Tuple[TopLeft, BottomRight]

class TemplateMatch:
    def __init__(self, template: Image):
        self.template = template
        self.t_height, self.t_width, *_ = template.image.shape

    @staticmethod
    def from_file(path: str) -> TemplateMatch:
        return TemplateMatch(np.array(PILImage.open(path)))

    def __SSD(self, image: np.ndarray) -> np.float64:
        '''
        SSD: Sum of Squared Difference
        in this case, it also know as 1D Euclidean distance
        '''
        assert (
            image.shape == self.template.image.shape
        ), "image and template must have the same shape"
        return np.sum((image - self.template.image) ** 2)
    
    def matching(self, image: Image) -> List[Marker]:
        markers: List[Marker] = []
        for i in range(image.image.shape[0] - self.t_height):
            for j in range(image.image.shape[1] - self.t_width):
                foucsed_image = image.image[i:i+self.t_height, j:j+self.t_width]
                if self.__SSD(foucsed_image) == 0:
                    markers.append(((j, i), (j+self.t_width, i+self.t_height)))

        return markers

    
