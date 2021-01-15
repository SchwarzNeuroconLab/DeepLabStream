"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

"""General use classes and functions for experiments"""



class InvalidSkeleton(Exception):
    """This exception catches invalid bodyparts (e.g., NaN) and returns them to the experiment.
    It's intended use is the continues run of an experiment even when incomplete skeletons are passed."""