# canary-decoder-esn-lstm

Source code (in Python) from the publication: Trouvain & Hinaut, ICANN 2021.

## Citation 
Trouvain, N., & Hinaut, X. (2021) Canary song decoder: Transduction and implicit segmentation with ESNs and LTSMs. In International Conference on Artificial Neural Networks (ICANN), pp. 71-82.

## Paper
Preprint version on HAL: https://hal.inria.fr/hal-03203374v2

Springer version: https://link.springer.com/chapter/10.1007/978-3-030-86383-8_6

## Abstract

Domestic canaries produce complex vocal patterns embedded in various levels of abstraction. Studying such temporal organization is of particular relevance to understand how animal brains represent and process vocal inputs such as language. However, this requires a large amount of annotated data. We propose a fast and easy-to-train transducer model based on RNN architectures to automate parts of the annotation process. This is similar to a speech recognition task. We demonstrate that RNN architectures can be efficiently applied on spectral features (MFCC) to annotate songs at time frame level and at phrase level. We achieved around 95% accuracy at frame level on particularly complex canary songs, and ESNs achieved around 5%
of word error rate (WER) at phrase level. Moreover, we are able to build this model using only around 13 to 20 min of annotated songs. Training time takes only 35 s using 2 h and 40 min of data for the ESN, allowing to quickly run experiments without the need of powerful hardware.
