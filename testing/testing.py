import torch
import numpy as np

from PIL import ImageGrab
from tkinter import *
from torchvision.transforms import ToTensor

root = Tk()
monitor_height = root.winfo_screenheight()
monitor_width = root.winfo_screenwidth()

tensorEncoder = ToTensor()

img = ImageGrab.grab((0,0,monitor_width,monitor_height))
img_RGB_array = tensorEncoder(img)
