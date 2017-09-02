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

#ifndef LAYERS_MULTILAYER_HPP
#define LAYERS_MULTILAYER_HPP

#include "TrainableLayer.hpp"
#include "../LayerFactory.hpp"

#include <unordered_map>

namespace layers {

    /******************************************************************************************//**
     * Represents a multi layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class MultiLayer : public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;
    private:
        std::string m_curFracSubType;
        std::string m_jsonMultiString;
        std::vector<std::string> m_layerNames;
        std::unordered_map<std::string, std::shared_ptr<TrainableLayer<TDevice> > > m_layersMap;
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        MultiLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
            const std::string        &layerType
            );

        /**
         * Destructs the Layer
         */
        virtual ~MultiLayer();

        /**
         * Returns the name of the current loaded layer
         *
         * @return The name of the current loaded layer
         */
        const std::string curName() const;

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * Returns the vector which contains the layer names
         *
         * @return The vector which contains the layer names
         */
        std::vector<std::string> &layerNames();

        /**
         * Returns the current weights
         *
         * @return The current weights
         */
        virtual real_vector& weights();

        /**
         * Returns the current weights
         *
         * @return The current weights
         */
        virtual const real_vector& weights() const;

        /**
         * Returns the current fraction subtype
         *
         * @return The current fraction subtype
         */
        std::string curFracSubType() const;

        /**
         * @see TrainableLayer::exportWeights()
         */
        virtual void exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator) const;

        /**
         * @see TrainableLayer::exportLayer()
         */
        virtual void exportLayer(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const;
    };

} // namespace layers


#endif // MULTILAYER
