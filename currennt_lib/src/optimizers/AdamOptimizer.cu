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

#include "AdamOptimizer.hpp"
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
        real_t beta1;
        real_t beta2;
        real_t beta3;
        real_t epsilon;
        real_t t;

        const real_t *weights;
        const real_t *weightUpdates;
        real_t       *mt;
        real_t       *vt;
        real_t       *pt;
        bool         nag;
        bool         average;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            real_t newWeight;
            if (nag)
                newWeight = _updateWeightNadamFn(weightIdx);
            else
                newWeight = _updateWeightAdamFn(weightIdx);
            if (average)
                return _averageFn(weightIdx, newWeight);
            else
                return newWeight;
        };

        // Adam
        __host__ __device__ real_t _updateWeightAdamFn (const int &weightIdx)
        {
            real_t nmt = beta1 * mt[weightIdx] + (1 - beta1) * weightUpdates[weightIdx];
            real_t nvt = beta2 * vt[weightIdx] + (1 - beta2) * weightUpdates[weightIdx] * weightUpdates[weightIdx];
            real_t mtc = nmt / (1 - helpers::TypedMath<real_t>::pow(beta1, t));  // These could be computed outside the kernel
            real_t vtc = nvt / (1 - helpers::TypedMath<real_t>::pow(beta2, t));  // But the penalty might be negigible
            mt[weightIdx] = nmt;
            vt[weightIdx] = nvt;
            real_t delta = - learningRate * mtc / (helpers::TypedMath<real_t>::sqrt(vtc) + epsilon);

            // return the new weight
            return (weights[weightIdx] + delta);
        };

        // Nadam
        // See "Dozat, Timothy, INCORPORATING NESTEROV MOMENTUM INTO ADAM"
        __host__ __device__ real_t _updateWeightNadamFn (const int &weightIdx)
        {
            real_t gt  = weightUpdates[weightIdx];
            real_t nmt = beta1 * mt[weightIdx] + (1 - beta1) * gt;
            real_t nvt = beta2 * vt[weightIdx] + (1 - beta2) * gt * gt;
            real_t mtc = beta1 * nmt / (1 - helpers::TypedMath<real_t>::pow(beta1, t+1))
                         + (1 - beta1) * gt / (1 - helpers::TypedMath<real_t>::pow(beta1, t));  // These could be computed outside the kernel
            real_t vtc = nvt / (1 - helpers::TypedMath<real_t>::pow(beta2, t));                 // But the penalty might be negigible
            mt[weightIdx] = nmt;
            vt[weightIdx] = nvt;
            real_t delta = - learningRate * mtc / (helpers::TypedMath<real_t>::sqrt(vtc) + epsilon);

            // return the new weight
            return (weights[weightIdx] + delta);
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
    void AdamOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateWeightFn updateWeightFn;

        updateWeightFn.beta1   = this->m_beta1;
        updateWeightFn.beta2   = this->m_beta2;
        updateWeightFn.epsilon = this->m_epsilon;
        updateWeightFn.nag     = this->m_nesterovMomentum;
        updateWeightFn.average = this->m_average;
        if (this->m_average)
            updateWeightFn.beta3   = this->m_beta3;

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

            updateWeightFn.mt = helpers::getRawPointer(layerParams->m_mts);
            updateWeightFn.vt = helpers::getRawPointer(layerParams->m_vts);
            updateWeightFn.pt = helpers::getRawPointer(layerParams->m_pts);
            updateWeightFn.t  = layerParams->m_t;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>((int)layer->weights().size()),
                layer->weights().begin(),
                updateWeightFn
                );
            ++(layerParams->m_t);
        }
    }

    template <typename TDevice>
    AdamOptimizer<TDevice>::AdamOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, 
        real_t learningRate, real_t beta1, real_t beta2, real_t epsilon, real_t beta3)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate (learningRate)
        , m_beta1        (beta1)
        , m_beta2        (beta2)
        , m_epsilon      (epsilon)
        , m_beta3        (beta3)
        , m_average      (beta3!=0)
    {
        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (layer) {
              	layers::MultiLayer<TDevice> *ml = dynamic_cast<layers::MultiLayer<TDevice>*>(layer);
                if (ml) {
                    for (auto& name: ml->layerNames()) {
                        auto lp = new LayerParams();
                        lp->m_t = 1;
                        lp->m_mts.resize(ml->weights().size(), 0);
                        lp->m_vts.resize(ml->weights().size(), 0);
                        lp->m_pts.resize(ml->weights().size(), 0);
                        m_paramsMap[name] = lp;
                    }
                } else {
                    auto lp = new LayerParams();
                    lp->m_t = 1;
                    lp->m_mts.resize(layer->weights().size(), 0);
                    lp->m_vts.resize(layer->weights().size(), 0);
                    lp->m_pts.resize(layer->weights().size(), 0);
                    m_paramsMap[layer->name()] = lp;
                }
            }
        }

        m_nesterovMomentum = Configuration::instance().nesterovMomentum();
    }

    template <typename TDevice>
    AdamOptimizer<TDevice>::~AdamOptimizer()
    {
    }

    template <typename TDevice>
    void AdamOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

//        Optimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void AdamOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

//        Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void AdamOptimizer<TDevice>::_updateParameters()
    {
        std::cout << "No improvement obtained, multiplying learning rate by " << Configuration::instance().learningRateDecay() << std::endl;
        real_t new_learningRate = m_learningRate * Configuration::instance().learningRateDecay();
        m_learningRate = max(Configuration::instance().minLearningRate(), new_learningRate);
    }

    // explicit template instantiations
    template class AdamOptimizer<Cpu>;
    template class AdamOptimizer<Gpu>;

} // namespace optimizers
