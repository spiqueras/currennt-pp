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

#include "TrainableLayer.hpp"
#include "MultiLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"

#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <typeinfo>

namespace layers {

    template <typename TDevice>
    MultiLayer<TDevice>::MultiLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer,
        const std::string &layerType)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0, precedingLayer)
    {
        // Obtain the names of the layers
        std::string multi = (*layerChild)["multi"].GetString();
        boost::algorithm::split(m_layerNames, multi, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
        
        if (m_layerNames.size() < 1)
            throw std::runtime_error(std::string("Error: bad multi name format '") + multi + "'");

        std::string layer_name = (*layerChild)["name"].GetString();
        bool is_first = true;
        this->m_curFracSubType = m_layerNames[0];
        for (auto& name : m_layerNames) {
            name = layer_name + "_" + name;
            (*layerChild)["name"].SetString(name.c_str());
          	Layer<TDevice> *layer = LayerFactory<TDevice>::createLayer(layerType, layerChild, weightsSection, 0, 0, &precedingLayer);        
            m_layersMap[name]     = std::shared_ptr<TrainableLayer<TDevice> >(dynamic_cast<TrainableLayer<TDevice>*>(layer));
            if (is_first) {
                this->outputs()      .resize(m_layersMap[name]->outputs().size());
                this->outputErrors() .resize(m_layersMap[name]->outputErrors().size());
                this->weightUpdates().resize(m_layersMap[name]->weightUpdates().size());
                is_first = false;
            }
        }

        m_jsonMultiString = m_layerNames[0].substr(this->name().length()+1, std::string::npos);
        for (int i = 1; i < m_layerNames.size(); ++i)
            m_jsonMultiString += "," + m_layerNames[i].substr(this->name().length()+1, std::string::npos);
    }

    template <typename TDevice>
    MultiLayer<TDevice>::~MultiLayer()
    {

    }

    template <typename TDevice>
    const std::string MultiLayer<TDevice>::curName() const
    {
        return this->name() + "_" + this->curFracSubType();
    }


    template <typename TDevice>
    const std::string& MultiLayer<TDevice>::type() const
    {
        return m_layersMap.begin()->second->type();
    }

    template <typename TDevice>
    void MultiLayer<TDevice>::computeForwardPass()
    {
        m_layersMap[curName()]->computeForwardPass();
        // Copy back the outputs
        thrust::copy(m_layersMap[curName()]->outputs().begin(),
                     m_layersMap[curName()]->outputs().end(),
                     this->outputs().begin());
    }

    template <typename TDevice>
    void MultiLayer<TDevice>::computeBackwardPass()
    {
        // Copy the outputErrors to the corresponding layer
        thrust::copy(this->outputErrors().begin(),
                     this->outputErrors().end(),
                     m_layersMap[curName()]->outputErrors().begin());
        m_layersMap[curName()]->computeBackwardPass();
        // Copy back the weightUpdates for SGD update
        thrust::copy(m_layersMap[curName()]->weightUpdates().begin(),
                     m_layersMap[curName()]->weightUpdates().end(),
                     this->weightUpdates().begin());

    }
    template <typename TDevice>
    void MultiLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction);
        m_curFracSubType = fraction.subType();
        m_layersMap[curName()]->loadSequences(fraction);
    }

    template <typename TDevice>
    std::vector<std::string>& MultiLayer<TDevice>::layerNames()
    {
        return m_layerNames;
    }

    template <typename TDevice>
    typename MultiLayer<TDevice>::real_vector& MultiLayer<TDevice>::weights()
    {
        return m_layersMap.at(curName())->weights();
    }

    template <typename TDevice>
    const typename MultiLayer<TDevice>::real_vector& MultiLayer<TDevice>::weights() const
    {
        return m_layersMap.at(curName())->weights();
    }

    template <typename TDevice>
    std::string MultiLayer<TDevice>::curFracSubType() const
    {
        return m_curFracSubType;
    }

    template <typename TDevice>
    void MultiLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator) const
    {
        for (auto& layerName : m_layerNames)
            m_layersMap.at(layerName)->exportWeights(weightsObject, allocator);
    }

    template <typename TDevice>
    void MultiLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("multi", m_jsonMultiString.c_str(), allocator);
    }

    template class MultiLayer<Cpu>;
    template class MultiLayer<Gpu>;
} // namespace layers
