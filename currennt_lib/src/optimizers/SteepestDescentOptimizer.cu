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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "SteepestDescentOptimizer.hpp"
#include "../Configuration.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/MultiLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"
#include "../helpers/TypedMath.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t momentum;
        real_t beta3;
        real_t t;

        const real_t *weights;
        const real_t *weightUpdates;
        real_t       *weightDeltas;
        real_t       *pt;
        bool         nag;
        bool         average;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            real_t newWeight;
            if (weightUpdates[weightIdx] == 0)
                return weights[weightIdx];
            if (nag)
                newWeight = _updateWeightNesterovFn(weightIdx);
            else
                newWeight = _updateWeightSGEFn(weightIdx);
            if (average)
                return _averageFn(weightIdx, newWeight);
            else
                return newWeight;
        };

        __host__ __device__ real_t _updateWeightSGEFn (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
            real_t newWeight = weights[weightIdx] + delta;

            return newWeight;
        };

        // Nesterov momentum
        // See "Bengio et al., ADVANCES IN OPTIMIZING RECURRENT NETWORKS"
        __host__ __device__ real_t _updateWeightNesterovFn (const int &weightIdx)
        {
            // calculate the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];

            // calculate the new weight
            real_t newWeight = (weights[weightIdx] - momentum * weightDeltas[weightIdx]) + (1 + momentum) * delta;

            // store the weight delta
            weightDeltas[weightIdx] = delta;

            return newWeight;
        };

        __host__ __device__ real_t _averageFn (const int &weightIdx, const real_t weight)
        {
            pt[weightIdx] = beta3 * pt[weightIdx] + (1 - beta3) * weight;
            return pt[weightIdx] / (1 - helpers::TypedMath<real_t>::pow(beta3, t));
        };
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.momentum     = m_momentum;
        updateWeightFn.nag          = this->m_nesterovMomentum;
        updateWeightFn.average      = this->m_average;
        if (this->m_average)
            updateWeightFn.beta3    = this->m_beta3;

//        if (true)
//             this->_injectGradientNoise(m_sigma);
        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            updateWeightFn.learningRate = m_learningRate;
            if (layer->learningRate() >= 0.0)
                updateWeightFn.learningRate = layer->learningRate();

            updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
            updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);

          	layers::MultiLayer<TDevice> *ml = dynamic_cast<layers::MultiLayer<TDevice>*>(layer);
            auto layerName   = ml ? ml->curName() : layer->name();
            auto layerParams = m_paramsMap[layerName];

            updateWeightFn.weightDeltas = helpers::getRawPointer(layerParams->m_weightDeltas);
            updateWeightFn.pt           = helpers::getRawPointer(layerParams->m_pts);
            updateWeightFn.t            = layerParams->m_t;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>((int)layer->weights().size()),
                layer->weights().begin(),
                updateWeightFn
                );
            ++(layerParams->m_t);
        }
//        m_sigma = 0.3 / helpers::TypedMath<real_t>::pow((1+m_t), 0.55);
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::SteepestDescentOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, 
        real_t learningRate, real_t momentum, real_t beta3)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate    (learningRate)
        , m_momentum        (momentum)
        , m_sigma           (1)
        , m_beta3           (beta3)
        , m_average         (beta3!=0)
    {
        // intialize the weight deltas vectors with zeros

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (layer) {
              	layers::MultiLayer<TDevice> *ml = dynamic_cast<layers::MultiLayer<TDevice>*>(layer);
                if (ml) {
                    for (auto& name: ml->layerNames()) {
                        auto lp = new LayerParams();
                        lp->m_t = 1;
                        lp->m_weightDeltas.resize(ml->weights().size(), 0);
                        lp->m_pts         .resize(ml->weights().size(), 0);
                        m_paramsMap[name] = lp;
                    }
                } else {
                    auto lp = new LayerParams();
                    lp->m_t = 1;
                    lp->m_weightDeltas.resize(layer->weights().size(), 0);
                    lp->m_pts         .resize(layer->weights().size(), 0);
                    m_paramsMap[layer->name()] = lp;
                }
            }
        }
        m_nesterovMomentum = Configuration::instance().nesterovMomentum();
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::~SteepestDescentOptimizer()
    {
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

  //      Optimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

   //     Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateParameters()
    {
        std::cout << "No improvement obtained, multiplying learning rate by " << Configuration::instance().learningRateDecay() << std::endl;
        real_t new_learningRate = m_learningRate * Configuration::instance().learningRateDecay();
        m_learningRate = max(Configuration::instance().minLearningRate(), new_learningRate);
    }

    // explicit template instantiations
    template class SteepestDescentOptimizer<Cpu>;
    template class SteepestDescentOptimizer<Gpu>;

} // namespace optimizers
