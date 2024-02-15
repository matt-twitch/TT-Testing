/*
  ==============================================================================

    TT_ADSR.h
    Created: 26 Jan 2024 11:06:34am
    Author:  Matt Twitchen

  ==============================================================================
*/

#pragma once
#include <tiny_dnn.h>
#include <JuceHeader.h>

using namespace tiny_dnn;

class TT_ADSRNetwork
{
public:
    
    TT_ADSRNetwork()
    {
        params.clip = 0;
    }
    
    ~TT_ADSRNetwork(){}
    
    void constructNetwork()
    {
        envNet << layers::fc(inputSize, numFeatures, false, backend);
        envNet << recurrent_layer(lstm(numFeatures, hiddenSize), numFeatures, params);
        envNet << activation::relu();
        envNet << layers::fc(hiddenSize, numFeatures, false, backend);
        envNet << activation::softmax();
    }
    
    void trainNetwork()
    {
        DBG("start training...");
        
        //cfNet.fit<mse>(opt, input, cutoff_values, batch_size, epochs);
        
        DBG("training ended");
    }
    
private:
    
    // network
    network<sequential> envNet;
    core::backend_t backend = core::default_engine();
    recurrent_layer_parameters params;
    const int inputSize = 4;
    const int numFeatures = 4;
    const int hiddenSize = 128;
    
    // training
    adam opt;
    size_t batchSize = 32;
    int epochs = 25;
};
