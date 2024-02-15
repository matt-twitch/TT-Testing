/*
  ==============================================================================

    TT_Cutoff.h
    Created: 26 Jan 2024 11:06:27am
    Author:  Matt Twitchen

  ==============================================================================
*/

#pragma once
#include <tiny_dnn.h>
#include <JuceHeader.h>

using namespace tiny_dnn;

class TT_CutoffNetwork
{
public:
    
    TT_CutoffNetwork()
    {
        
    }
    
    ~TT_CutoffNetwork(){}
    
    void constructNetwork()
    {
        cfNet << layers::fc(1, numFeatures, false, backend);
        cfNet << layers::fc(numFeatures, hiddenSize, false, backend);
        cfNet << activation::relu();
        cfNet << layers::fc(hiddenSize, numFeatures, false, backend);
        cfNet << activation::sigmoid();
    }
    
    void trainNetwork()
    {
        DBG("start training...");
        
        //cfNet.fit<mse>(opt, input, cutoff_values, batch_size, epochs);
        
        DBG("training ended");
    }
    
private:
    
    // network
    network<sequential> cfNet;
    core::backend_t backend = core::default_engine();
    const int inputSize = 1;
    const int numFeatures = 1;
    const int hiddenSize = 128;
    
    // training
    adam opt;
    size_t batchSize = 32;
    int epochs = 25;
};
