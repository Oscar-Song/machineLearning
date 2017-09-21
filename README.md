# ProDx Machine Learning Anomaly Detector

A system of data-processing and neural network to detect anomalies in
messages from ACL TOP 550 and to output results as graphs on the
web browser.

##Overview

  - Display the anomaly level for each area in a general log in graphs
  - Display the graphs on a web browser
  - Determine the status of each area as okay, monitoring, or To be Attended
  - Also allows
    - Train the neural network with different set of data
    - Apply to different ACL TOP model if retrained

## Motivation
  The creation of this project is to improve the efficiency, applicability, reliability, and flexibility of ProDx data-processing that was done by the existing program. The pre-existing data-flow, implemented using VBA on Macro Excel in 2006, has not been updated for a decade and often lead to high false-positives due to human hard-coded thresholds. This project, inspired by machine-learning methodology, specifically an autoencoder neural network, use the pattern from big data to detect anomalies in a general log. It is also implemented using python, which makes advantage of many python libraries and dependencies as well as the object-oriented structure.

## Get Started
### Dependencies
First make sure that those dependencies are installed
  - [python 2.7] - To run python and use python libraries such as Numpy
  - [theano 0.9] - The neural network library to run the
autoencoder
  - [panda] - A handy library for data-filtering
  - [xlrd] - Excel sheet manipulations
  - [glob] - A package for finding path name and directory
  - [cPickle] - Saver and loader of autoencoder parementers
    
    Those dependencies can all be installed using pip

### Prerequisite

Make sure that you have the following files in your directory:

* driver.py
* genLogToTable.py
* dataFilter.py
* autoencoder.py
as well as
* golden_stat.csv
* area folder
* Error Code List - ACL TOP Logbook Viewer Rev 3 with Pre-
Analytical Warning and Error Codes.xlsx

those files are necessary for training
* createGolden.py
* trainAutoencoder.py

### Running

Assume the neural network is trained. Simply install the pacage.

```sh
$ cd prodxDetector
$ npm install -d
```

To open the GUI

```sh
$ python driver.py
```

(Optional) Only for training
First have a folder name ‘data’ that has all the good general log files in the current directory.

```sh
$ python create_CSVs.py
$ python train_autoencoder.py
```

This will create 8 text files storing trained parameters of autoencoders for 8 areas respectively. When running driver.py again, those text files will be loaded in automatically.

## Authors

Oscar Song - Scrum master and contributer to major module assembly and implementations.
Janelle Lines - Contributer to neural network design, implementation, optimization, and testing
Sunjay Ravishankqr - Contributer to data conversion and status computation modules
Oshin Mundada - Contributer to data retrival for autoencoder training and front-end developing.

##License
This project is supervised and owned by Instrumentation Laboratory.


