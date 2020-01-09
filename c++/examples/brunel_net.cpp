/*
Copyright (C) 2020 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "neuralgpu.h"

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " n_neurons\n";
    return 0;
  }
  int arg1;
  sscanf(argv[1], "%d", &arg1);
  NeuralGPU neural_gpu;
  cout << "Building ...\n";

  neural_gpu.SetMaxSpikeBufferSize(10); // max spike buffer size per neuron
  
  int n_receptors = 2;

  int order = arg1/5;
  int NE = 4 * order;      // number of excitatory neurons
  int NI = 1 * order;      // number of inhibitory neurons
  int n_neurons = NE + NI; // number of neurons in total

  int CE = 800;  // number of excitatory synapses per neuron
  int CI = CE/4;  // number of inhibitory synapses per neuron

  float Wex = 0.04995;
  float Win = 0.35;

  // each host has a poisson generator
  float poiss_rate = 20000.0; // poisson signal rate in Hz
  float poiss_weight = 0.369;
  float poiss_delay = 0.2; // poisson signal delay in ms
  int n_pg = n_neurons; // number of poisson generators
  // create poisson generator
  int pg = neural_gpu.CreatePoissonGenerator(n_pg, poiss_rate);

  // each host has n_neurons neurons with n_receptor receptor ports
  int neuron = neural_gpu.CreateNeuron("AEIF", n_neurons, n_receptors);
  int exc_neuron = neuron;      // excitatory neuron id
  int inh_neuron = neuron + NE; // inhibitory neuron id
  
  // the following parameters are set to the same values on all hosts
  float E_rev[] = {0.0, -85.0};
  float taus_decay[] = {1.0, 1.0};
  float taus_rise[] = {1.0, 1.0};
  neural_gpu.SetNeuronVectParams("E_rev", neuron, n_neurons, E_rev, 2);
  neural_gpu.SetNeuronVectParams("taus_decay", neuron, n_neurons,
				 taus_decay, 2);
  neural_gpu.SetNeuronVectParams("taus_rise", neuron, n_neurons, taus_rise, 2);
  
  float mean_delay = 0.5;
  float std_delay = 0.25;
  float min_delay = 0.1;
  // Excitatory connections
  // connect excitatory neurons to port 0 of all neurons
  // normally distributed delays, weight Wex and CE connections per neuron
  float *exc_delays = neural_gpu.RandomNormalClipped(CE*n_neurons, mean_delay,
  						     std_delay, min_delay,
  						     mean_delay+3*std_delay);
  float *exc_weights = new float[CE*n_neurons];
  for (int i=0; i<CE*n_neurons; i++) exc_weights[i] = Wex;
  
  neural_gpu.ConnectFixedIndegreeArray(exc_neuron, NE, neuron, n_neurons,
				  0, exc_weights, exc_delays, CE);
  delete[] exc_delays;
  delete[] exc_weights;

  // Inhibitory connections
  // connect inhibitory neurons to port 1 of all neurons
  // normally distributed delays, weight Win and CI connections per neuron
  float *inh_delays = neural_gpu.RandomNormalClipped(CI*n_neurons, mean_delay,
  						     std_delay, min_delay,
  						     mean_delay+3*std_delay);
  float *inh_weights = new float[CI*n_neurons];
  for (int i=0; i<CI*n_neurons; i++) inh_weights[i] = Win;
  
  neural_gpu.ConnectFixedIndegreeArray(inh_neuron, NI, neuron, n_neurons,
				  1, inh_weights, inh_delays, CI);

  delete[] inh_delays;
  delete[] inh_weights;

  // connect poisson generator to port 0 of all neurons
  neural_gpu.ConnectOneToOne(pg, neuron, n_neurons, 0, poiss_weight,
				  poiss_delay);
  
  char filename[] = "test_brunel_net.dat";
  
  int i_neuron_arr[] = {neuron, neuron+rand()%n_neurons,
		     neuron+n_neurons-1}; // any set of neuron indexes
  // create multimeter record of V_m
  std::string var_name_arr[] = {"V_m", "V_m", "V_m"};
  neural_gpu.CreateRecord(string(filename), var_name_arr, i_neuron_arr, 3);

  neural_gpu.SetRandomSeed(1234ULL); // just to have same results in different simulations
  neural_gpu.Simulate();

  neural_gpu.MpiFinalize();

  return 0;
}