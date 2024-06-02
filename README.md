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

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
