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

#include "RnnLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

//    typedef activation_functions::Logistic cell_output_act_fn_t;

    template <typename TActFn>
    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char *patTypes;

        const real_t *ogBiasWeights;

        real_t *ogActs;

        __host__ __device__ real_t operator() (const int &outputIdx, const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations
            real_t ogAct = ogActs[outputIdx];

            // add bias activations
            ogAct += bias * ogBiasWeights[blockIdx];

            ogActs[outputIdx] = TActFn::fn(ogAct);

            // calculate the block output
            real_t output = ogActs[outputIdx];

            return output;
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    template <typename TActFn>
    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char *patTypes;

        const real_t *ogActs;

        real_t *ogDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   checkPatType = t.get<2>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    ogDeltas[outputIdx] = 0;
                    return;
                }
            }

            // load the cell activation
            real_t ogAct = ogActs[outputIdx];

            // calculate the output gate delta
            real_t ogDelta = TActFn::deriv(ogAct) * outputErr;

            // store the delta
            ogDeltas[outputIdx] = helpers::limitedError(ogDelta);
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;   
        const real_t *bwOutputs;
        const real_t *fwOgDeltas;  
        const real_t *bwOgDeltas;  

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            // weightType = 0b00XX
            // weightType = 0b0000 ( 0): OG input weight
            //              0b0001 ( 1): OG bias weight
            //              0b0010 ( 2): OG internal weight
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;

            int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 1 * biwc);

            int weightTypeX = weightType & 0x3;

            // calculate indices, offsets and increments 
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightTypeX) {
            // input weight
            case 0x0: 
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = inputWeightIdx / precLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;
                    
                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;

            // bias weight
            case 0x1: 
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            case 0x2: 
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = internalWeightIdx / effLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *niagDeltasLut[] = {
                fwOgDeltas,
                bwOgDeltas
            };

            // calculate the weight update over all patterns            
            const real_t *offDeltas = &niagDeltasLut[(isBwStateWeight ? 1 : 0)][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;
                    
                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }

            return wu;
        }
    };
    
} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice, typename TActFn>
    RnnLayer<TDevice, TActFn>::RnnLayer(const helpers::JsonValue &layerChild, 
                                  const helpers::JsonValue &weightsSection,
                                  Layer<TDevice> &precedingLayer,
                                  bool bidirectional)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 1, (bidirectional ? 0.5 : 1) * helpers::safeJsonGetInt(layerChild, "size"), precedingLayer)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");

        // set raw pointers
        int ls  = this->size();
        int pls = this->precedingLayer().size();

        _rawOgBiasWeights     = helpers::getRawPointer(this->weights()) + ls * pls;

        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            int pls = this->precedingLayer().size();
            int ls  = this->size();
            int els = this->size() / (m_isBidirectional ? 2 : 1);

            // cell states, niags, deltas, ...
            Cpu::real_vector tmp(this->outputs().size() / (m_isBidirectional ? 2 : 1), 0);

            if (m_isBidirectional) {
                fwbw->tmpOutputs      = tmp;
                fwbw->tmpOutputErrors = tmp;
            }
            else {
                fwbw->tmpOutputs     .swap(this->_outputs());
                fwbw->tmpOutputErrors.swap(this->outputErrors());
            }

            fwbw->ogActs   = tmp;
            fwbw->ogDeltas = tmp;

            // weight matrices
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                real_vector       *wts = wtsArr[wmArrIdx];

                int numInputWeights      = ls * pls;
                int numInternalWeights   = ls * els;
                int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
                int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) + (ls * (pls + 1));

                wm->ogInput    = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart);
                wm->ogInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart);
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows   = this->size() / (m_isBidirectional ? 2 : 1);
                int cols   = this->parallelSequences();
                int offset = timestep * rows * cols;

                timestep_matrices_t tm;
                tm.tmpOutputs      = helpers::Matrix<TDevice>(&fwbw->tmpOutputs,      rows, cols, offset);
                tm.tmpOutputErrors = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors, rows, cols, offset);
                tm.ogActs          = helpers::Matrix<TDevice>(&fwbw->ogActs,          rows, cols, offset);
                tm.ogDeltas        = helpers::Matrix<TDevice>(&fwbw->ogDeltas,        rows, cols, offset);

                fwbw->timestepMatrices.push_back(tm);
            }
        }

        if (!m_isBidirectional) {
            m_fw.tmpOutputs     .swap(this->_outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
    }

    template <typename TDevice, typename TActFn>
    RnnLayer<TDevice, TActFn>::~RnnLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    const std::string& RnnLayer<TDevice, TActFn>::type() const
    {
        static std::string s;

        if (s.empty()) {
            if (m_isBidirectional) {
                if (typeid(TActFn) == typeid(activation_functions::Tanh))
                    s = "brnn_tanh";
                else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                    s = "brnn_logistic";
                else if (typeid(TActFn) == typeid(activation_functions::Identity))
                    s = "brnn_identity";
                else if (typeid(TActFn) == typeid(activation_functions::Relu))
                    s = "brnn_relu";
                else
                    throw std::runtime_error("Unsupported activation function");
            } else {
                if (typeid(TActFn) == typeid(activation_functions::Tanh))
                    s = "rnn_tanh";
                else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                    s = "rnn_logistic";
                else if (typeid(TActFn) == typeid(activation_functions::Identity))
                    s = "rnn_identity";
                else if (typeid(TActFn) == typeid(activation_functions::Relu))
                    s = "rnn_relu";
                else
                    throw std::runtime_error("Unsupported activation function");
            }
        }
        return s;
    }

    template <typename TDevice, typename TActFn>
    bool RnnLayer<TDevice, TActFn>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice, typename TActFn>
    const typename TDevice::real_vector& RnnLayer<TDevice, TActFn>::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }

    template <typename TDevice, typename TActFn>
    const typename TDevice::real_vector& RnnLayer<TDevice, TActFn>::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }

    template <typename TDevice, typename TActFn>
    void RnnLayer<TDevice, TActFn>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction);

        m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();

            fwbw->ogActsMatrix   = helpers::Matrix<TDevice>(&fwbw->ogActs,   rows, cols);
            fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
        }
    }

    template <typename TDevice, typename TActFn>
    void RnnLayer<TDevice, TActFn>::computeForwardPass()
    {
        // for unidirectional RNN, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional)
            m_fw.tmpOutputs.swap(this->_outputs());

        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.ogActsMatrix.assignProduct(m_fw.weightMatrices.ogInput, true, m_precLayerOutputsMatrix, false);

            // backward states
            if (m_isBidirectional)
                m_bw.ogActsMatrix.assignProduct(m_bw.weightMatrices.ogInput, true, m_precLayerOutputsMatrix, false);
        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockOutputFn<TActFn> fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.bias               = this->bias();
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.ogBiasWeights      = _rawOgBiasWeights;
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect outputs from previous timestep
                if (timestep != 0)
                    m_fw.timestepMatrices[timestep].ogActs.addProduct(m_fw.weightMatrices.ogInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);

                // compute outputs
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(
                           thrust::make_tuple(thrust::constant_iterator<bool>(!timestep),
                                              thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_fw.tmpOutputs.begin() + n*timestep,
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.ogBiasWeights     += els;
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);

                for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1)
                        m_bw.timestepMatrices[timestep].ogActs.addProduct(m_bw.weightMatrices.ogInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(
                               thrust::make_tuple(thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),
                                                  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        m_bw.tmpOutputs.begin() + n*timestep,
                        fn
                        );
                }
            }
        }}

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
            fn.layerSize    = this->size();
            fn.effLayerSize = this->size() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                this->_outputs().begin(),
                fn
                );
        }
        else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }
    }

    template <typename TDevice, typename TActFn>
    void RnnLayer<TDevice, TActFn>::computeBackwardPass()
    {
        // for unidirectional RNN, we can write the output errors directly in the layer output errors vector
        if (m_isBidirectional) {
            internal::ResortOutputErrorsFn fn;
            fn.layerSize      = this->size();
            fn.effLayerSize   = this->size() / 2;
            fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
            fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }
        else {
            m_fw.tmpOutputs     .swap(this->outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }

        // calculate the block errors
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockErrorsFn<TActFn> fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);
            fn.ogDeltas           = helpers::getRawPointer(m_fw.ogDeltas);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1)
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.ogInternal, false, m_fw.timestepMatrices[timestep+1].ogDeltas, false);

                // compute errors
                thrust::for_each(
                    thrust::make_zip_iterator(
                           thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep,
                                              thrust::counting_iterator<int>(n*timestep),
                                              thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))
                    ),
                    thrust::make_zip_iterator(
                           thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep+n,
                                              thrust::counting_iterator<int>(n*timestep)+n,
                                              thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)
                    ),
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);
                fn.ogDeltas           = helpers::getRawPointer(m_bw.ogDeltas);

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0)
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.ogInternal, false, m_bw.timestepMatrices[timestep-1].ogDeltas, false);

                    // compute errors
                    thrust::for_each(
                        thrust::make_zip_iterator(
                               thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep,
                                                  thrust::counting_iterator<int>(n*timestep),
                                                  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))
                        ),
                        thrust::make_zip_iterator(
                               thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep+n,
                                                  thrust::counting_iterator<int>(n*timestep)+n,
                                                  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)
                        ),
                        fn
                        );
                }
            }
        }}

        // back-propagate the error to the preceding layer
        {{
            TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.ogInput, false, m_fw.ogDeltasMatrix, false);

                // backward states
                if (m_isBidirectional)
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ogInput, false, m_bw.ogDeltasMatrix, false);
            }
        }}

        // compute the weight updates
        {{
            internal::ComputeWeightUpdateFn fn;
            fn.layerSize             = this->size();
            fn.effLayerSize          = this->size() / (m_isBidirectional ? 2 : 1);
            fn.precLayerSize         = this->precedingLayer().size();
            fn.timestepDistance      = this->parallelSequences() * this->size() / (m_isBidirectional ? 2 : 1);
            fn.parallelSequences     = this->parallelSequences();
            fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
            fn.biasWeightsOffset     = this->size() * this->precedingLayer().size();
            fn.internalWeightsOffset = fn.biasWeightsOffset + this->size();
            fn.bias                  = this->bias();
            fn.plOutputs             = helpers::getRawPointer(this->precedingLayer().outputs());
            fn.fwOutputs             = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs             = helpers::getRawPointer(m_bw.tmpOutputs);
            fn.fwOgDeltas            = helpers::getRawPointer(m_fw.ogDeltas);
            fn.bwOgDeltas            = helpers::getRawPointer(m_bw.ogDeltas);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + (int)this->weightUpdates().size(),
                this->_weightUpdates().begin(),
                fn
                );
        }}

        // re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
            this->outputErrors().swap(m_fw.tmpOutputErrors);
            this->_outputs()    .swap(m_fw.tmpOutputs);
        }
    }


    // explicit template instantiations
    template class RnnLayer<Cpu, activation_functions::Tanh>;
    template class RnnLayer<Gpu, activation_functions::Tanh>;
    template class RnnLayer<Cpu, activation_functions::Logistic>;
    template class RnnLayer<Gpu, activation_functions::Logistic>;
    template class RnnLayer<Cpu, activation_functions::Identity>;
    template class RnnLayer<Gpu, activation_functions::Identity>;
    template class RnnLayer<Cpu, activation_functions::Relu>;
    template class RnnLayer<Gpu, activation_functions::Relu>;

} // namespace layers
