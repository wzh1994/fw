import ctypes
import sys
from glob import glob
import os

dl = ctypes.CDLL('../x64/Release/fw.dll')
dl.ShowDialog()

# dl = ctypes.CDLL('play.dll')
# dl.play()
