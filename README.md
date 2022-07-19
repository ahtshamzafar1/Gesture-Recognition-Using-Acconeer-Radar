# Gesture Recognition Using Acconeer Radar
This repository provides a complete python based setup for gesture recognition. 

# LSTM_for_Gesture_Recognition
This file provides the complete LSTM model built in keras/tensorflow.  
The real-time HGR system involves continuous streaming of data. Since the sweeps are correlated and prediction depends on a stream of multiple sweeps instead of just one sweep, the LSTM network is also a considerable option for the HGR system implementation.
LSTM’s working is a bit different in the sense that it has a global state which is maintained among all the inputs. All the previous input’s context is transferred to future inputs by a global state. And because of this nature, it doesn’t suffer from vanishing and exploding gradient problems. LSTM networks are generally used for time series data analysis and for the kind of data that is dependent on previous data, so having the memory feature in LSTM can help in those scenarios.