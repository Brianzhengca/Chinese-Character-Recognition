import sys
sys.path.insert(0, "OCR")
sys.path.insert(0, "Correction")
from OCR import segmentpredict
from Correction import correct
reference_dt = { #reference dictionary for chinese character and their different parts
    "始":"女台",
    "女台":"始"
}
class Main:
    def __init__(self):
        self.corrector = correct.generate_correction()
    def predict(self, path):
        self.segment_predict = segmentpredict.Segment_Predict()
        self.predicted_results = self.segment_predict.segment_predict(path)
        return self.generate_correction(self.predicted_results)
    def generate_correction(self, inp):
        if self.corrector.correct(inp) != []:
            for index, char in enumerate(inp):
                if (index < len(inp) - 1):
                    first = char
                    second = inp[index+1]
                    if first+second in reference_dt.keys():
                        inp[index] = reference_dt[first+second]
                        inp.pop(index+1)
        return ''.join(inp)
flow = Main()
print("Prediction Result:", flow.predict("OCR/segmentation_test/test.jpg"))