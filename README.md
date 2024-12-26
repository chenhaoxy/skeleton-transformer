# Meta-process Driven 3-D Skeleton Feature Learning for Enhanced Human Action Recognition

## Abstruct
Directly inputting a video that stores a long behavior process into the recognition model will greatly increase the computational cost of the model, and some redundant information will also affect the recognition accuracy of the model. In order to solve these problems, a classification method of human behavior based on meta-process learning is proposed. This method establishes a mechanism for judging the core joint points of the human skeleton during movement and a mechanism for dividing the meta-process of the movement process based on the movement data of the core joint points. At the same time, a kinematic meta-process is described in the form of a special Euclidean group, and a method for calculating specific parameters based on Transformer network is established. In addition, a method is designed to characterize the common external features of a class of motions by transforming probability matrices, and the classifier is trained by fusing the data of internal features and external features of the meta-process.

## how to use
```
python main_multipart_ntu.py --config config/nturgbd-cross-subject/lst_joint.yaml
```

## Dataset https://rose1.ntu.edu.sg/dataset/actionRecognition
