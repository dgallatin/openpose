"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python
"""
import numpy as np
import ctypes as ct
import os
from sys import platform
dir_path = os.path.dirname(os.path.realpath(__file__))


class RECT(ct.Structure):
    _fields_ = [("x", ct.c_float),
                ("y", ct.c_float),
                ("width", ct.c_float),
                ("height", ct.c_float)]


class OpenPoseFace(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop = np.ctypeslib.load_library('_openposeface', dir_path+'/_openposeface.so')
    elif platform == "darwin":
        _libop = np.ctypeslib.load_library('_openposeface', dir_path+'/_openposeface.dylib')
    elif platform == "win32":
        try:
            _libop = np.ctypeslib.load_library('_openposeface', dir_path+'/Release/_openposeface.dll')
        except OSError as e:
            _libop = np.ctypeslib.load_library('_openposeface', dir_path+'/Debug/_openposeface.dll')
    _libop.newOPFace.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_char_p]
    _libop.newOPFace.restype = ct.c_void_p
    _libop.delOPFace.argtypes = [ct.c_void_p]
    _libop.delOPFace.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        ct.POINTER(RECT), ct.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32)]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
                                    self.encode(params["net_output_size"]),
                                    self.encode(params["net_input_size"]),
                                    params["num_gpu_start"],
                                    self.encode(params["default_model_folder"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, face_rectangles):
        """
        Forward: Takes in an image and returns the human 2D face landmarks

        Parameters
        ----------
        image : color image of type ndarray
        face_rectangles : bounding boxes of faces in the image of type list[RECT]

        Returns
        -------
        array: ndarray of human 2D poses [People * FaceLandmark * XYConfidence]
        """
        shape = image.shape
        size = np.zeros(shape=(3), dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], ct.byref(face_rectangles), len(face_rectangles),size)
        array = np.zeros(shape=(size), dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array
