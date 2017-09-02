/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "LayerFactory.hpp"

#include "layers/InputLayer.hpp"
#include "layers/FeedForwardLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/LstmLayer.hpp"
#include "layers/RnnLayer.hpp"
#include "layers/MultiLayer.hpp"
#include "layers/SsePostOutputLayer.hpp"
#include "layers/NormSsePostOutputLayer.hpp"
#include "layers/AbsPostOutputLayer.hpp"
#include "layers/RmsePostOutputLayer.hpp"
#include "layers/WeightedRmsePostOutputLayer.hpp"
#include "layers/CePostOutputLayer.hpp"
#include "layers/SseMaskPostOutputLayer.hpp"
#include "layers/WeightedSsePostOutputLayer.hpp"
#include "layers/BinaryClassificationLayer.hpp"
#include "layers/MulticlassClassificationLayer.hpp"
#include "activation_functions/Tanh.cuh"
#include "activation_functions/Logistic.cuh"
#include "activation_functions/Identity.cuh"
#include "activation_functions/Relu.cuh"

#include <stdexcept>


template <typename TDevice>
layers::Layer<TDevice>* LayerFactory<TDevice>::createLayer(
		const std::string &layerType, const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection, int parallelSequences, 
        int maxSeqLength, layers::Layer<TDevice> *precedingLayer,
        const std::string &underlyingLayerType)
{
    using namespace layers;
    using namespace activation_functions;

    if (layerType == "input")
    	return new InputLayer<TDevice>(layerChild, parallelSequences, maxSeqLength);
    else if (layerType == "feedforward_tanh")
    	return new FeedForwardLayer<TDevice, Tanh>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "feedforward_logistic")
    	return new FeedForwardLayer<TDevice, Logistic>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "feedforward_identity")
    	return new FeedForwardLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "feedforward_relu")
    	return new FeedForwardLayer<TDevice, Relu>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "softmax")
    	return new SoftmaxLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "lstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "blstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "rnn_tanh")
    	return new RnnLayer<TDevice, Tanh>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "brnn_tanh")
    	return new RnnLayer<TDevice, Tanh>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "rnn_logistic")
    	return new RnnLayer<TDevice, Logistic>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "brnn_logistic")
    	return new RnnLayer<TDevice, Logistic>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "rnn_identity")
    	return new RnnLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "brnn_identity")
    	return new RnnLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "rnn_relu")
    	return new RnnLayer<TDevice, Relu>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "brnn_relu")
    	return new RnnLayer<TDevice, Relu>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "multi")
    	return new MultiLayer<TDevice>(layerChild, weightsSection, *precedingLayer, underlyingLayerType);
    else if (layerType == "sse" || layerType == "normsse"  || layerType == "weightedsse" || layerType == "abs" || layerType == "rmse" || layerType == "wrmse" || layerType == "ce" || layerType == "wf" || layerType == "binary_classification" || layerType == "multiclass_classification") {
        if (layerType == "sse")
    	    return new SsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        if (layerType == "normsse")
    	    return new NormSsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "weightedsse")
    	    return new WeightedSsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "abs")
            return new AbsPostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "rmse")
            return new RmsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "wrmse")
    	    return new WeightedRmsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "ce")
            return new CePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        if (layerType == "sse_mask" || layerType == "wf") // wf provided for compat. with dev. version
    	    return new SseMaskPostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "binary_classification")
    	    return new BinaryClassificationLayer<TDevice>(layerChild, *precedingLayer);
        else // if (layerType == "multiclass_classification")
    	    return new MulticlassClassificationLayer<TDevice>(layerChild, *precedingLayer);
    }
    else
        throw std::runtime_error(std::string("Unknown layer type '") + layerType + "'");
}


// explicit template instantiations
template class LayerFactory<Cpu>;
template class LayerFactory<Gpu>;
