# An Attempt to Improve Chinese Character Recognition with KenLM

Inspired by [This paper](https://aclanthology.org/2024.lt4hala-1.14.pdf).

## Description

The goal of the project is as follows: Given an image of a line of Chinese text, correctly identify the characters 

The original idea of this project stemmed from reading papers about using language models to improve the accuracy of Optical Character Recognition (OCR). Currently, many OCR techniques rely on simple
image recognition without understanding the meaning behind the text it is trying to recognize. This can lead to many errors. Therefore it is natural to think that language models can be used to 
improve results post-recognition. 

Specifically, Chinese characters can sometimes be composed of two distinct but perfectly valid characters. For example, the character "好" can be split into "女" and "子" and the character "始" can be split
into "女" and "台". For someone with bad handwriting like me, normal OCR programs might incorrectly split such characters into two. Upon further investigations, I realized simple segmentors such as OpenCV's findContours function might 
erroneously segment "准备开始" into "准备开女台". My idea is to create a reference map that maps these characters to their split characters, and then use [KenLM](https://kheafield.com/code/kenlm/) to guess the right version of the recognized sentence.  

In this project, I trained a simple CNN on the [CASIA-HWDB](https://ieeexplore.ieee.org/document/6065272) dataset to recognize images of individual handwritten Chinese characters. I used OpenCV-Python's findContour() function
to segment the image of a line of text into images of individual characters. Finally, I used a pre-trained KenLM model from [pycorrector](https://github.com/shibing624/pycorrector?tab=readme-ov-file) to check for errors 
in the recognized characters and replace them according to my reference map if necessary. 


## Getting Started

### Dependencies

* MacOS Sonoma 14.3
* Python 3.12.1
* Keras 3.3.3
* OpenCV-Python 4.9.0.80
* NumPy 1.26.4
* Pillow 10.2.0
* Pandas 2.2.0
* Matplotlib 3.8.3
* PyCorrector 1.0.4

### Usage with Pre-trained CNN model

```
git clone https://github.com/Brianzhengca/Chinese-Character-Recognition.git
cd Chinese-Character-Recognition
python3 main.py
```

### Usage with Training CNN Yourself

* It is a little complicated this way since the CASIA-HWDB dataset is too big to be uploaded to GitHub. 
* Essentially you need to download the dataset and place it inside a directory called ```archive``` under ```/OCR/Data Preprocessing```.
* The file ```/OCR/Data Preprocessing/main.py``` must be able to access ```OCR/Data Preprocessing/archive/CASIA-HWDB_Train/Train``` and ```OCR/Data Preprocessing/archive/CASIA-HWDB_Test/Test```
* Afterwards run the following command
```
cd Chinese-Character-Recognition
source run.sh
```

## Limitations

Because of limited resource constraints, I only trained the CNN to recognize the following characters ```大家准备开始女台``` and my reference map only consists of ```{"始":"女台","女台":"始"}```. I expect the performance to significantly decrease if it was trained to recognize more characters. This project is mainly an implementation of a fun idea and is not intended for any use beyond that. 

## License

Use it however you like. 

## Acknowledgments

* All the papers and projects mentioned above in the description
* Stackoverflow
* Kaggle
