/*
 *  diffusion_connection.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef diffusion_connection_H
#define diffusion_connection_H
#include <cmath>

/* BeginUserDocs: synapse, instantaneous rate

Short description
+++++++++++++++++

Synapse type for instantaneous rate connections between neurons of type siegert_neuron

Description
+++++++++++

diffusion_connection is a connector to create
instantaneous connections between neurons of type siegert_neuron. The
connection type is identical to type rate_connection_instantaneous
for instantaneous rate connections except for the two parameters
drift_factor and diffusion_factor substituting the parameter weight.

These two factor origin from the mean-field reduction of networks of
leaky-integrate-and-fire neurons. In this reduction the input to the
neurons is characterized by its mean and its variance. The mean is
obtained by a sum over presynaptic activities (e.g as in eq.28 in
[1]_), where each term of the sum consists of the presynaptic activity
multiplied with the drift_factor. Similarly, the variance is obtained
by a sum over presynaptic activities (e.g as in eq.29 in [1]_), where
each term of the sum consists of the presynaptic activity multiplied
with the diffusion_factor. Note that in general the drift and
diffusion factors might differ from the ones given in eq. 28 and 29.,
for example in case of a reduction on the single neuron level or in
case of distributed in-degrees (see discussion in chapter 5.2 of [1]_)

The values of the parameters delay and weight are ignored for
connections of this type.

Transmits
+++++++++

DiffusionConnectionEvent

References
++++++++++


.. [1] Hahne J, Dahmen D, Schuecker J, Frommer A,
       Bolten M, Helias M, Diesmann, M. (2017).
       Integration of continuous-time dynamics in a
       spiking neural network simulator.
       Frontiers in Neuroinformatics, 11:34.
       DOI: https://doi.org/10.3389/fninf.2017.00034


See also
++++++++

siegert_neuron

EndUserDocs */

namespace diffusion_connection_ns
{
  enum ParamIndexes {
    i_drift_factor = 0, i_diffusion_factor,
    N_PARAM
  };

  const std::string diffusion_connection_param_name[N_PARAM] = {
    "drift_factor", "diffusion_factor"
  };

  __device__ __forceinline__ void diffusion_connection_Update(float *param)
  {
    //printf("Dt: %f\n", Dt);
    double drift_factor = param[i_drift_factor];
    double diffusion_factor = param[i_diffusion_factor];
  }
}


#endif
