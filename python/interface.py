import ctypes
import os
import torch
from torch import multiprocessing

class FireworkTypeEnum(object):
    def __init__(self):
        self.__dict__.update(dict(
            Normal=0,
            MultiExplosion=1,
            Strafe=2,
            Circle=3,
            Twinkle=4,
            DualMixture=5,
            TriplicateMixture=6,
            NormalCircleAndTwinkle=7,
            FiveNormal=8,
            SixCircle=9,
            ThreeNormalAndTwinkle=10,
            ThreeNormalCircleTwinkle=11
        ))


FireworkType = FireworkTypeEnum()


class CxxInterface(object):
    def __init__(self):
        self.mfc_dll = ctypes.CDLL('fw.dll')
        self.play_dll = ctypes.CDLL('play.dll')
        self.extract_dll = ctypes.CDLL('FrameDifference.dll')

    # 返回修改之后的args
    def call_mfc(self, fw_type, args, movie_name):
        if isinstance(args, tuple):
            args = list(args)
        assert isinstance(args, list)
        arg_array = ctypes.c_float * len(args)
        c_args = arg_array(*args)

        movie_name = os.path.join('movies', movie_name)
        if isinstance(movie_name, str):
            movie_name = movie_name.encode(encoding="utf-8")
        assert isinstance(movie_name, bytes)
        length = len(movie_name)
        movie_name = ctypes.c_char_p(movie_name)

        self.mfc_dll.ShowDialog(fw_type, c_args, movie_name, length)
        return list(c_args)

    def play(self, fw_type, args):
        if isinstance(args, tuple):
            args = list(args)
        assert isinstance(args, list)
        arg_array = ctypes.c_float * len(args)
        c_args = arg_array(*args)
        self.play_dll.play(fw_type, c_args)

    def play_and_save(self, fw_type, args, movie_name):
        if isinstance(args, tuple):
            args = list(args)
        assert isinstance(args, list)
        arg_array = ctypes.c_float * len(args)
        c_args = arg_array(*args)
        if isinstance(movie_name, str):
            movie_name = movie_name.encode(encoding="utf-8")
        assert isinstance(movie_name, bytes)
        length = len(movie_name)
        movie_name = ctypes.c_char_p(movie_name)
        self.play_dll.playAndSave(fw_type, c_args, movie_name, length)

    def extract(self, fname):
        fname = os.path.join('movies', fname)
        if isinstance(fname, str):
            fname = fname.encode(encoding="utf-8")
        assert isinstance(fname, bytes)
        length = len(fname)
        fname = ctypes.c_char_p(fname)
        self.extract_dll.extract(fname, length)


class Interface(object):
    def __init__(self):
        pass

    def _call_mfc(self, fw_type, args, movie_name, q1, q2):
        r = CxxInterface().call_mfc(fw_type, args, movie_name)
        q1.put(torch.tensor(r))
        q2.get()

    def call_mfc(self, fw_type, args, movie_name):
        q1 = multiprocessing.Queue()
        q2 = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._call_mfc, args=(fw_type, args, movie_name, q1, q2))
        p.start()
        r = list(q1.get().numpy())
        q2.put(0)
        p.join()
        return r

    def _play(self, fw_type, args):
         CxxInterface().play(fw_type, args)

    def play(self, fw_type, args):
        p = multiprocessing.Process(target=self._play, args=(fw_type, args))
        p.start()
        p.join()

    def _play_and_save(self, fw_type, args, movie_name):
        CxxInterface().play_and_save(fw_type, args, movie_name)

    def play_and_save(self, fw_type, args, movie_name):
        p = multiprocessing.Process(target=self._play_and_save, args=(fw_type, args, movie_name))
        p.start()
        p.join()

    def extract(self, fname):
         CxxInterface().extract(fname)


interface = Interface()
