from __future__ import print_function, division, unicode_literals, absolute_import

from .statis import STATIS, ANISOSTATIS, STATISData
from .wine_data import get_wine_data, get_sup_wine_data

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
