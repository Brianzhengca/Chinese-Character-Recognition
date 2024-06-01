import keras
import numpy as np
from PIL import Image, ImageEnhance

class Predictor:
    def __init__(self):
        self.model = keras.saving.load_model("model.keras")
    def predict(self, path, encoded_mapping):
        self.target_path = path
        self.encoded_mapping = encoded_mapping
        self.image = Image.open(self.target_path)
        self.image = self.image.convert('L')
        #print(self.image.size[0], self.image.size[1])
        self.image = self.image.resize((28,28), Image.LANCZOS)
        self.enh = ImageEnhance.Contrast(self.image)
        self.enh_factor = np.random.uniform(2,3,1)
        self.image = self.enh.enhance(factor=self.enh_factor)
        self.inp_arr = np.array(self.image)
        self.inp_arr = np.reshape(self.inp_arr,(1, 28, 28))
        #self.image.show()
        self.prediction = self.model.predict(self.inp_arr)[0] #returns numpy array like array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prediction = list(self.prediction) #convert to list
        print(self.prediction)
        return self.encoded_mapping[self.prediction.index(1)] #match encoder mapping
