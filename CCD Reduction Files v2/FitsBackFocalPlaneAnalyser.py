# -*- coding: utf-8 -*-
from CCDBackFocalPlaneAnalyser import CCDBackFocalPlaneAnalyser
from FitsReducer import FitsReducer
from errors import FitsBackFocalPlaneAnalyserError


class FitsBackFocalPlaneAnalyser(CCDBackFocalPlaneAnalyser):

    @staticmethod
    def _loadfile(filename, masterpath, savepath):
        f =  FitsReducer(filename, masterpath, savepath)
        return f.data
