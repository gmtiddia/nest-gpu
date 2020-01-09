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
  NeuralGPU ngpu;
  cout << "Building ...\n";
  
  srand(12345);
  int n_neurons = 10000;
  
  // each host has n_neurons neurons with 1 receptor ports
  int neuron = ngpu.CreateNeuron("AEIF", n_neurons, 1);

  // the following parameters are set to the same values on all hosts
  ngpu.SetNeuronParams("a", neuron, n_neurons,  4.0);
  ngpu.SetNeuronParams("b", neuron, n_neurons,  80.5);
  ngpu.SetNeuronParams("E_L", neuron, n_neurons,  -70.6);
  ngpu.SetNeuronParams("I_e", neuron, n_neurons,  700.0);

  string filename = "test_constcurr.dat";
  int i_neurons[] = {rand()%n_neurons}; // any set of neuron indexes
  string var_name[] = {"V_m"};

  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neurons, 1);

  ngpu.Simulate();

  return 0;
}