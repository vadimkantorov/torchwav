Read and write .WAV files in PyTorch without any dependencies.

This is an adaptation of [scipy.io.wavfile](https://github.com/scipy/scipy/blob/dc0bb8b/scipy/io/wavfile.py). The API is the same except that `mmap = True` is not supported. Reading directly from file to an array (via `fromfile`) is also not supported.

Please refer or original docs for [read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) and [write](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html).

**TODO**: fix byte_order, byteswap


## Tests
### Create sample audio test.wav
```shell
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 test.wav
``` 

### Check equivalence with scipy.io.wavfile
```python
import torchwavfile
import scipy.io.wavfile

sample_rate_torch, data_torch = torchwavfile.read('test.wav')
sample_rate_scipy, data_scipy = scipy.io.wavfile.read('test.wav')
assert sample_rate_torch == sample_rate_scipy and (data_torch.numpy() == data_scipy).all()

torchwavfile.write('test_torch.wav', sample_rate_torch, data_torch)
scipy.io.wavfile.write('test_scipy.wav', sample_rate_scipy, data_scipy)
assert open('test_torch.wav', 'rb').read() == open('test_scipy.wav', 'rb').read()
``` 
