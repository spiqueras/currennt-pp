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

#ifndef OPTIMIZERS_ADAMOPTIMIZER_HPP
#define OPTIMIZERS_ADAMOPTIMIZER_HPP

#include "Optimizer.hpp"

#include <vector>
#include <unordered_map>


namespace optimizers {

    /******************************************************************************************//**
     * Optimizer that uses steepest descent
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class AdamOptimizer : public Optimizer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

        struct LayerParams
        {
            unsigned m_t;
            real_vector m_mts;
            real_vector m_vts;
            real_vector m_pts;
        };

    private:
        real_t m_learningRate;
        const real_t m_beta1;
        const real_t m_beta2;
        const real_t m_beta3;
        const real_t m_epsilon;
        bool m_nesterovMomentum;
        bool m_average;

        std::unordered_map<std::string, LayerParams*> m_paramsMap;
        
    protected:
        virtual void _updateWeights();
        virtual void _updateParameters();

    public:
        /**
         * Constructs the optimizer
         *
         * @param neuralNetwork   The neural network to operate on
         * @param trainingSet     The set of training sequences
         * @param validationSet   The set of validation sequences
         * @param testSet         The set of test sequences
         * @param maxEpochs       The maximum total number of epochs to train
         * @param maxEpochsNoBest The number of epochs in which no new lowest error could be
         *                        achieved before training is stopped
         * @param validateEvery   After how many epochs the validation error shall be calculated
         * @param testEvery       After how many epochs the test error shall be calculated
         * @param learningRate    The learning rate
         * @param momentum        The momentum
         */
        AdamOptimizer(
            NeuralNetwork<TDevice> &neuralNetwork,
            data_sets::DataSet     &trainingSet,
            data_sets::DataSet     &validationSet,
            data_sets::DataSet     &testSet,
            int maxEpochs, 
            int maxEpochsNoBest, 
            int validateEvery,
            int testEvery,
            real_t learningRate,
            real_t beta1,
            real_t beta2,
            real_t epsilon,
            real_t beta3
            );

        /**
         * Destructs the optimizer
         */
        virtual ~AdamOptimizer();

        /**
         * @see Optimizer::exportState
         */
        virtual void exportState(const helpers::JsonDocument &jsonDoc) const;

        /**
         * @see Optimizer::importState
         */
        virtual void importState(const helpers::JsonDocument &jsonDoc);

    };

} // namespace optimizers


#endif // OPTIMIZERS_ADAMOPTIMIZER_HPP
