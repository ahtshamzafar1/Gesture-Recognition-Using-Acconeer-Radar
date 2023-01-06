# Gesture Recognition Using Acconeer Radar
# Collected DATASET Available on KAGGLE. 
https://www.kaggle.com/datasets/ahtshamzafar/mmwave-radar-dataset-for-hand-gesture-recognition

This repository provides a complete python based setup for gesture recognition. 

# LSTM_for_Gesture_Recognition
This file provides the complete LSTM model built in keras/tensorflow.  
The real-time HGR system involves continuous streaming of data. Since the sweeps are correlated and prediction depends on a stream of multiple sweeps instead of just one sweep, the LSTM network is also a considerable option for the HGR system implementation.
LSTM’s working is a bit different in the sense that it has a global state which is maintained among all the inputs. All the previous input’s context is transferred to future inputs by a global state. And because of this nature, it doesn’t suffer from vanishing and exploding gradient problems. LSTM networks are generally used for time series data analysis and for the kind of data that is dependent on previous data, so having the memory feature in LSTM can help in those scenarios.

# CNN_For_Gesture_Recognition
This methodology utilizes Deep Convolutional Neural networks (DCNNs). The DCNN is composed of various layers such as the convolutional layer, activation layer, pooling layer, dropout layer, and dense layer. Convolutional filter trains to find different features in an input feature map. The nonlinear activation function performs a non-linear transformation on the output of the convolutional layer which assists to categorize different classes. There are a variety of activation functions available such as SoftMax, Sigmoid, Exponential linear Unit, and Rectifier linear units(ReLU). ReLU is one of the most used activation functions and also it solves the vanishing gradient problem. This network uses the ReLU activation.

# Data_Acquisition
This file utilizes the base code provided by Acconeer and develops on top of it to acquire specific dataset samples that will be later used for training the neurlal network.

# Calibration
This file utilizes the base code provided by Acconeer and develops on top of it to calibrate the samples that will be later used for training the neurlal network. To remove static background noise based on wherever
the radar is placed, 50 sweeps were recorded containing only background information. While recording background noise no other activity was performed near the sensor. The aver- aged sweep vector represents the static background noise that is removed from every sweep vector during data set collection and model testing.
