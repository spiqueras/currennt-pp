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

#ifndef LAYERS_RNNLAYER_HPP
#define LAYERS_RNNLAYER_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a fully connected RNN layer
     *
     * weights; with P = precedingLayer().size() and L = size():
     *    ~ weights from preceding layer:
     *        - [0 .. PL-1]:    output gate
     *    ~ bias weights:
     *        - [PL + 0  .. PL + L-1]:  output gate
     *    ~ internal weights (from other cells in the same layer):
     *        - [(P+1)L + 0   .. (P+1)L + LL-1]:  output gate
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function (tanh, logistic, identity or relu)
     *********************************************************************************************/
    template <typename TDevice, typename TActFn>
    class RnnLayer : public TrainableLayer<TDevice>
    {

    private:
        real_t *_rawOgBiasWeights;

        helpers::Matrix<TDevice> m_precLayerOutputsMatrix;

    public:
        typedef typename TDevice::real_vector real_vector;
        struct weight_matrices_t {
            helpers::Matrix<TDevice> ogInput;
            helpers::Matrix<TDevice> ogInternal;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpOutputErrors;
            helpers::Matrix<TDevice> ogActs;
            helpers::Matrix<TDevice> ogDeltas;
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpOutputErrors;
            real_vector ogActs;
            real_vector ogDeltas;

            helpers::Matrix<TDevice> ogActsMatrix;
            helpers::Matrix<TDevice> ogDeltasMatrix;

            weight_matrices_t                weightMatrices;
            weight_matrices_t                weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;
        };

        const bool m_isBidirectional;
        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        RnnLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
            bool                      bidirectional = false
            );

        /**
         * Destructs the Layer
         */
        virtual ~RnnLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * Returns true if the layer is bidirectional
         *
         * @return True if the layer is bidirectional
         */
        bool isBidirectional() const;

        /**
         * Returns the output gate activations
         *
         * @return The output gate activations
         */
        const real_vector& outputGateActs() const;

        /**
         * Returns the output gate deltas
         *
         * @return The output gate deltas
         */
        const real_vector& outputGateDeltas() const;

        /**
         * @see Layer::loadSequences
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_RNNLAYER_HPP
