import math
import wave
import struct

noise_output = wave.open('noise2.wav', 'w')
noise_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))

values = []
frequency = 440
rate = 44100
samples = [float(math.sin(2.0 * math.pi * frequency * t / rate)) for t in xrange(0, int(0.2 * rate))]
print samples
sm = []
amplitude = 32767
for i in range(0, len(samples)):
        packed_value = struct.pack('h', int(amplitude*samples[i]))
        sm.append(int(amplitude*samples[i]))
        values.append(packed_value)
        values.append(packed_value)

print values
value_str = ''.join(values)
noise_output.writeframes(value_str)
noise_output.close()