from pycorrector import Corrector
import json

class generate_correction():
    def __init__(self):
        self.corrector = Corrector()
    def correct(self, sentence):
        return self.corrector.detect(''.join(sentence)) == []
