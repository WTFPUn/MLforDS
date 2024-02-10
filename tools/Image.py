from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from tools.GaussianBlur import GaussianBlur
from tools.TemplateMatch import TemplateMatch
from PIL import Image as PILImage
import PIL
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt

Coord = Tuple[int, int]
TopLeft = Coord
BottomRight = Coord

Marker = Tuple[TopLeft, BottomRight]

ImageType = Literal["L", "RGB"]

class Image:
  def __init__(self, image: np.ndarray):
    self.image = image

  
  def display(self):
    plt.imshow(self.image, cmap='gray')
    plt.show()
  
  @staticmethod
  def from_file(path: str, mode: ImageType = "RGB") -> Image:
    return Image(np.array(PILImage.open(path).convert(mode)))

  def gaussian_blur(self, kernel_size: int, sigma: int):
    gau = GaussianBlur(kernel_size, sigma)
    self.image = gau.filter(self.image, gau.kernel)
  
  def addPadding(self, padding: int) -> None:
    copy_img = np.copy(self.image)
    self.image = np.pad(copy_img, ((padding,padding),(padding,padding)), 'constant', constant_values=0)

  def __sub__(self, other: Image) -> Image:
    return Image(self.image - other.image)
  
  def __add__(self, other: Image) -> Image:
    return Image(self.image + other.image)
  
  def __mul__(self, other: Image) -> Image:
    return Image(self.image * other.image)
  
  def __truediv__(self, other: Image) -> Image:
    return Image(self.image / other.image)
  
  def __pow__(self, other: Image) -> Image:
    return Image(self.image ** other.image)
  
  def __eq__(self, other: Image) -> Image:
    return Image(self.image == other.image)
  
  def __ne__(self, other: Image) -> Image:
    return Image(self.image != other.image)
  
  def __gt__(self, other: Image) -> Image:
    return Image(self.image > other.image)
  
  def __lt__(self, other: Image) -> Image:
    return Image(self.image < other.image)
  
  def __ge__(self, other: Image) -> Image:
    return Image(self.image >= other.image)
  
  def __le__(self, other: Image) -> Image:
    return Image(self.image <= other.image)

  def __and__(self, other: Image) -> Image:
    return Image(self.image & other.image)
  
  def __or__(self, other: Image) -> Image:
    return Image(self.image | other.image)

  def templateMatching(self, pattern: Image) -> List[Marker]:
    '''
    pattern: np.ndarray of shape (n, m)

    this method will recieve a pattern and return a list of markers (top_left, bottom_right) of the pattern in the image.
    '''
    template = TemplateMatch(pattern)
    return template.matching(self)
  
  def draw_rectangle(self, marks: List[Marker], color: Literal["r", "g", "b"]):
    if self.image.ndim == 2:
        rgb_img = np.stack((self.image,)*3, axis=-1)
    else:
        rgb_img = self.image
  

    fig, ax = plt.subplots()
    ax.imshow(rgb_img)

    for mark in marks:
        top_left, bottom_right = mark

        bottom_left = (top_left[0], bottom_right[1])
        print(top_left, bottom_right)
        rect = Rectangle(bottom_left, bottom_right[0]-bottom_left[0], top_left[1]-bottom_left[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()




