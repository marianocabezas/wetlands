import time
import os
import re
import sys
import traceback
from functools import reduce
from scipy import ndimage as nd
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.ndimage.morphology import binary_erosion as imerode
import torch


"""
Utility functions
"""


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCII codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
        'clr': '\033[K',
    }
    return codes


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(
        filter(
            lambda d: os.path.isdir(os.path.join(path, d)),
            os.listdir(path)
        )
    )
    # Patients used during training
    return patient_paths


def print_message(message):
    """
    Function to print a message with a custom specification
    :param message: Message to be printed.
    :return: None.
    """
    c = color_codes()
    dashes = ''.join(['-'] * (len(message) + 11))
    print(dashes)
    print(
        '%s[%s]%s %s' %
        (c['c'], time.strftime("%H:%M:%S", time.localtime()), c['nc'], message)
    )
    print(dashes)


def time_f(f, stdout=None, stderr=None):
    """
    Function to time another function.
    :param f: Function to be run. If the function has any parameters, it should
    be passed using the lambda keyword.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return: The result of running f.
    """
    # Init
    stdout_copy = sys.stdout
    if stdout is not None:
        sys.stdout = stdout

    start_t = time.time()
    try:
        ret = f()
    except Exception as e:
        ret = None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('{0}: {1}'.format(type(e).__name__, e), file=stderr)
        traceback.print_tb(exc_traceback, file=stderr)
    finally:
        if stdout is not None:
            sys.stdout = stdout_copy

    print(
        time.strftime(
            'Time elapsed = %H hours %M minutes %S seconds',
            time.gmtime(time.time() - start_t)
        )
    )
    return ret


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


def get_int(string):
    """
    Function to get the int number contained in a string. If there are more
    than one int number (or there is a floating point number), this function
    will concatenate all digits and return an int, anyways.
    :param string: String that contains an int number
    :return: int number
    """
    return int(''.join(filter(str.isdigit, string)))


def normalise(image):
    """
    Function to normalise an image given its mean and standard deviation
     (z-score).
    :param image:  Image to normalise.
    :return:
    """
    im_mean = np.mean(image, axis=(1, 2), keepdims=True)
    im_std = np.std(image, axis=(1, 2), keepdims=True)

    return (image - im_mean) / im_std
