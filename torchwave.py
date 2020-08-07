import ctypes
import wave
import torch

def read(filename):
    with wave.open(filename, 'r') as w:
        fs = w.getframerate()
        tensor, storage = (torch.ShortTensor, torch.ShortStorage) if w.getsampwidth() == 2 else (torch.CharTensor, torch.CharStorage)
        frames = w.readframes(w.getnframes())
        data = tensor(storage.from_buffer(frames, byte_order = 'native')).reshape(-1, w.getnchannels())
        return fs, data.squeeze(1)

def write(filename, rate, data):
    assert data.is_contiguous() and not data.is_floating_point()
    itemsize = torch.iinfo(data.dtype).bits // 8
    with wave.open(filename, 'w') as w:
        w.setframerate(rate)
        w.setnchannels(data.shape[1] if data.ndim == 2 else 1)
        w.setnframes(data.shape[0])
        w.setsampwidth(itemsize)
        w.writeframes(ctypes.cast(data.data_ptr(), ctypes.POINTER(ctypes.c_ubyte * (data.numel() * itemsize))).contents)
