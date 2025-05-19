import cv2
import numpy as np
import matplotlib.pyplot as plt
#import tkinter as tk
#from tkinter import filedialog
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.draw import line
from collections import deque
import heapq
