Read and write .WAV files in PyTorch without any dependencies via two methods:
- wrapper of Python [`wave`](https://docs.python.org/3/library/wave.html) core module in [torchwave.py](./torchwave.py)
- adaptation of [`scipy.io.wavfile`](https://github.com/scipy/scipy/blob/dc0bb8b/scipy/io/wavfile.py) in [torchwavefile.py](./torchwavefile.py)

API for both methods follow the original `scipy.io.wavfile`, except that `mmap = True` is not supported. Reading directly from file to an array (via `fromfile`) is also not supported. Big-endian platforms are also not supported / tested. Please refer to original docs for [read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) and [write](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html).

### Create sample audio test.wav
```shell
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 test.wav
``` 

### Check equivalence with scipy.io.wavfile
```python
import scipy.io.wavfile
import torchwavfile
import torchwave

sample_rate_scipy, data_scipy = scipy.io.wavfile.read('test.wav')
sample_rate_torch, data_torch = torchwavfile.read('test.wav')
sample_rate_wave, data_wave = torchwave.read('test.wav')
assert sample_rate_torch == sample_rate_scipy and (data_torch.numpy() == data_scipy).all()
assert sample_rate_wave == sample_rate_scipy and (data_wave.numpy() == data_scipy).all()

scipy.io.wavfile.write('test_scipy.wav', sample_rate_scipy, data_scipy)
torchwavfile.write('test_torch.wav', sample_rate_torch, data_torch)
torchwave.write('test_wave.wav', sample_rate_wave, data_wave)
assert open('test_torch.wav', 'rb').read() == open('test_scipy.wav', 'rb').read()
assert open('test_wave.wav', 'rb').read() == open('test_scipy.wav', 'rb').read()
``` 
