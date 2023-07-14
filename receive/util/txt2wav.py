import sys
import wave
import os
import numpy as np
import matplotlib.pyplot as plt


# arg='20211216142612.txt'
#
# pcmdata=np.loadtxt(arg, dtype='short')
#
#
# with wave.open('test.wav', 'wb') as wavfile:
#     wavfile.setparams((1, 2, 48000,w 0, 'NONE', 'NONE'))
#     wavfile.writeframes(pcmdata.tobytes())
#     wavfile.close()

with wave.open('../1-1.wav', 'rb') as f:
    params = f.getparams()
    print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
res=np.frombuffer(str_data, dtype='short')
print(res)
np.savetxt("test.txt", res)

from scipy.io import wavfile
samplerate, data = wavfile.read('../1-1.wav')
print(data)