from array import array
from pygame.mixer import Sound, get_init, pre_init
import copy
import pygame
from time import sleep
import math

class Note(Sound):

    def __init__(self, frequency, volume=100):
        self.frequency = frequency
        Sound.__init__(self, buffer=self.build_samples())
        self.set_volume(volume)

    def build_samples(self):
        period = int(round(get_init()[0] / self.frequency))
        samples = array("h", [0] * period)
        #rate = 44100
        #x = [float(math.sin(2.0 * math.pi * self.frequency * t / rate)) for t in xrange(0, int(0.2 * rate))]
        #samples = array("h", x)
        amplitude = 2 ** (abs(get_init()[1]) - 1) - 1

        for time in xrange(period):
            if time < period / 2:
                samples[time] = amplitude
            else:
                samples[time] = -amplitude
        print samples
        return samples


pre_init(44100, -16, 1, 1024)
pygame.init()
Note(440).play(-1)
print "played"
sleep(5)
