##################################
import torch
import ctypes
def frombuffer(bytes, dtype):
    dtype2tensor = dict(i2 = torch.ShortTensor, f4 = torch.FloatTensor, u8 = torch.CharTensor)
    dtype2storage = dict(i2 = torch.ShortStorage, f4 = torch.FloatStorage, u8 = torch.CharStorage)
    byte_order = {'<' : 'little', '>' : 'big'}.get(dtype[0], 'native')
    dtype = dtype.strip('<=>')
    return dtype2tensor[dtype](dtype2storage[dtype].from_buffer(bytes, byte_order = byte_order))
def tobytes(tensor):
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_ubyte * nbytes(tensor))).contents
def kind(tensor):
    integer = not tensor.is_floating_point()
    signed = tensor.is_signed()
    return 'i' if integer and signed else 'u' if integer and not signed else 'f'
def itemsize(tensor):
    return (torch.finfo if tensor.is_floating_point() else torch.iinfo)(tensor.dtype).bits // 8
def nbytes(tensor):
    return tensor.numel() * itemsize(tensor)
def byteorder(tensor):
    return '='
##################################

import io
import sys
import torch
import struct
import warnings
from enum import IntEnum


__all__ = [
    'WavFileWarning',
    'read',
    'write'
]


class WavFileWarning(UserWarning):
    pass


class WAVE_FORMAT(IntEnum):
    """
    WAVE form wFormatTag IDs

    Complete list is in mmreg.h in Windows 10 SDK.  ALAC and OPUS are the
    newest additions, in v10.0.14393 2016-07
    """
    UNKNOWN = 0x0000
    PCM = 0x0001
    IEEE_FLOAT = 0x0003
    EXTENSIBLE = 0xFFFE

KNOWN_WAVE_FORMATS = {WAVE_FORMAT.PCM, WAVE_FORMAT.IEEE_FLOAT}


def _read_fmt_chunk(fid, is_big_endian):
    """
    Returns
    -------
    size : int
        size of format subchunk in bytes (minus 8 for "fmt " and itself)
    format_tag : int
        PCM, float, or compressed format
    channels : int
        number of channels
    fs : int
        sampling frequency in samples per second
    bytes_per_second : int
        overall byte rate for the file
    block_align : int
        bytes per sample, including all channels
    bit_depth : int
        bits per sample

    Notes
    -----
    Assumes file pointer is immediately after the 'fmt ' id
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    size = struct.unpack(fmt+'I', fid.read(4))[0]

    if size < 16:
        raise ValueError("Binary structure of wave file is not compliant")

    res = struct.unpack(fmt+'HHIIHH', fid.read(16))
    bytes_read = 16

    format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res

    if format_tag == WAVE_FORMAT.EXTENSIBLE and size >= (16+2):
        ext_chunk_size = struct.unpack(fmt+'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2+4:2+4+16]
            # GUID template {XXXXXXXX-0000-0010-8000-00AA00389B71} (RFC-2361)
            # MS GUID byte order: first three groups are native byte order,
            # rest is Big Endian
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt+'I', raw_guid[:4])[0]
        else:
            raise ValueError("Binary structure of wave file is not compliant")

    if format_tag not in KNOWN_WAVE_FORMATS:
        try:
            format_name = WAVE_FORMAT(format_tag).name
        except ValueError:
            format_name = f'{format_tag:#06x}'
        raise ValueError(f"Unknown wave file format: {format_name}. Supported "
                         "formats: " +
                         ', '.join(x.name for x in KNOWN_WAVE_FORMATS))

    # move file pointer to next chunk
    if size > bytes_read:
        fid.read(size - bytes_read)

    # fmt should always be 16, 18 or 40, but handle it just in case
    _handle_pad_byte(fid, size)

    return (size, format_tag, channels, fs, bytes_per_second, block_align,
            bit_depth)


def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian,
                     block_align, mmap=False):
    """
    Notes
    -----
    Assumes file pointer is immediately after the 'data' id

    It's possible to not use all available bits in a container, or to store
    samples in a container bigger than necessary, so bytes_per_sample uses
    the actual reported container size (nBlockAlign / nChannels).  Real-world
    examples:

    Adobe Audition's "24-bit packed int (type 1, 20-bit)"

        nChannels = 2, nBlockAlign = 6, wBitsPerSample = 20

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-int12-AFsp.wav
    is:

        nChannels = 2, nBlockAlign = 4, wBitsPerSample = 12

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf
    gives an example of:

        nChannels = 2, nBlockAlign = 8, wBitsPerSample = 20
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    # Size of the data subchunk in bytes
    size = struct.unpack(fmt+'I', fid.read(4))[0]

    # Number of bytes per sample (sample container size)
    bytes_per_sample = block_align // channels
    n_samples = size // bytes_per_sample
    if bit_depth == 8:
        dtype = 'u1'
    else:
        if format_tag == WAVE_FORMAT.PCM:
            dtype = f'{fmt}i{bytes_per_sample}'
        else:
            dtype = f'{fmt}f{bytes_per_sample}'

    start = fid.tell()
    if not mmap:
        #try:
        #    data = fromfile(fid, dtype=dtype, count=n_samples)
        #except io.UnsupportedOperation:  # not a C-like file
            fid.seek(start, 0)  # just in case it seeked, though it shouldn't
            data = frombuffer(fid.read(size), dtype=dtype)
    else:
        assert not mmap, 'mmap is not supported'
        #data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start,
        #                    shape=(n_samples,))
        #fid.seek(start + size)

    _handle_pad_byte(fid, size)

    if channels > 1:
        data = data.reshape(-1, channels)
    return data


def _skip_unknown_chunk(fid, is_big_endian):
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'

    data = fid.read(4)
    # call unpack() and seek() only if we have really read data from file
    # otherwise empty read at the end of the file would trigger
    # unnecessary exception at unpack() call
    # in case data equals somehow to 0, there is no need for seek() anyway
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)
        _handle_pad_byte(fid, size)


def _read_riff_chunk(fid):
    str1 = fid.read(4)  # File signature
    if str1 == b'RIFF':
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_big_endian = True
        fmt = '>I'
    else:
        # There are also .wav files with "FFIR" or "XFIR" signatures?
        raise ValueError(f"File format {repr(str1)} not understood. Only "
                         "'RIFF' and 'RIFX' supported.")

    # Size of entire file
    file_size = struct.unpack(fmt, fid.read(4))[0] + 8

    str2 = fid.read(4)
    if str2 != b'WAVE':
        raise ValueError(f"Not a WAV file. RIFF form type is {repr(str2)}.")

    return file_size, is_big_endian


def _handle_pad_byte(fid, size):
    # "If the chunk size is an odd number of bytes, a pad byte with value zero
    # is written after ckData." So we need to seek past this after each chunk.
    if size % 2:
        fid.seek(1, 1)


def read(filename, mmap=False):
    assert mmap is False, 'mmap argument is not supported'
                             
    if hasattr(filename, 'read'):
        fid = filename
        mmap = False
    else:
        fid = open(filename, 'rb')

    try:
        file_size, is_big_endian = _read_riff_chunk(fid)
        fmt_chunk_received = False
        data_chunk_received = False
        while fid.tell() < file_size:
            # read the next chunk
            chunk_id = fid.read(4)

            if not chunk_id:
                if data_chunk_received:
                    # End of file but data successfully read
                    warnings.warn(
                        "Reached EOF prematurely; finished at {:d} bytes, "
                        "expected {:d} bytes from header."
                        .format(fid.tell(), file_size),
                        WavFileWarning, stacklevel=2)
                    break
                else:
                    raise ValueError("Unexpected end of file.")
            elif len(chunk_id) < 4:
                msg = f"Incomplete chunk ID: {repr(chunk_id)}"
                # If we have the data, ignore the broken chunk
                if fmt_chunk_received and data_chunk_received:
                    warnings.warn(msg + ", ignoring it.", WavFileWarning,
                                  stacklevel=2)
                else:
                    raise ValueError(msg)

            if chunk_id == b'fmt ':
                fmt_chunk_received = True
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                format_tag, channels, fs = fmt_chunk[1:4]
                bit_depth = fmt_chunk[6]
                block_align = fmt_chunk[5]
                if bit_depth not in {8, 16, 32, 64, 96, 128}:
                    raise ValueError("Unsupported bit depth: the wav file "
                                     "has {}-bit data.".format(bit_depth))
            elif chunk_id == b'fact':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id == b'data':
                data_chunk_received = True
                if not fmt_chunk_received:
                    raise ValueError("No fmt chunk before data")
                data = _read_data_chunk(fid, format_tag, channels, bit_depth,
                                        is_big_endian, block_align, mmap)
            elif chunk_id == b'LIST':
                # Someday this could be handled properly but for now skip it
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id in {b'JUNK', b'Fake'}:
                # Skip alignment chunks without warning
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                              WavFileWarning, stacklevel=2)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        if not hasattr(filename, 'read'):
            fid.close()
        else:
            fid.seek(0)

    return fs, data


def write(filename, rate, data):
    data = data.contiguous()
    if hasattr(filename, 'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')

    fs = rate

    try:
        dkind = kind(data)
        if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and
                                                 itemsize(data) == 1)):
            raise ValueError("Unsupported data type '%s'" % data.dtype)

        header_data = b''

        header_data += b'RIFF'
        header_data += b'\x00\x00\x00\x00'
        header_data += b'WAVE'

        # fmt chunk
        header_data += b'fmt '
        if dkind == 'f':
            format_tag = WAVE_FORMAT.IEEE_FLOAT
        else:
            format_tag = WAVE_FORMAT.PCM
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
        bit_depth = itemsize(data) * 8
        bytes_per_second = fs*(bit_depth // 8)*channels
        block_align = channels * (bit_depth // 8)

        fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs,
                                     bytes_per_second, block_align, bit_depth)
        if not (dkind == 'i' or dkind == 'u'):
            # add cbSize field for non-PCM files
            fmt_chunk_data += b'\x00\x00'

        header_data += struct.pack('<I', len(fmt_chunk_data))
        header_data += fmt_chunk_data

        # fact chunk (non-PCM files)
        if not (dkind == 'i' or dkind == 'u'):
            header_data += b'fact'
            header_data += struct.pack('<II', 4, data.shape[0])

        # check data size (needs to be immediately before the data chunk)
        if ((len(header_data)-4-4) + (4+4+nbytes(data))) > 0xFFFFFFFF:
            raise ValueError("Data exceeds wave file size limit")

        fid.write(header_data)

        # data chunk
        fid.write(b'data')
        fid.write(struct.pack('<I', nbytes(data)))
        if byteorder(data) == '>' or (byteorder(data) == '=' and
                                           sys.byteorder == 'big'):
            assert False, 'big-endian is not supported'
            data = data.byteswap()
        _array_tofile(fid, data)

        # Determine file size and place it in correct
        #  position at start of the file.
        size = fid.tell()
        fid.seek(4)
        fid.write(struct.pack('<I', size-8))

    finally:
        if not hasattr(filename, 'write'):
            fid.close()
        else:
            fid.seek(0)


def _array_tofile(fid, data):
    # ravel gives a c-contiguous buffer
    fid.write(tobytes(data))
