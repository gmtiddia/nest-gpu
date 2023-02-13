import sys
import nest
import numpy as np
tolerance = 0.0005
neuron = nest.Create('aeif_cond_alpha', 1)
nest.SetStatus(neuron, {"V_peak": 0.0, "a": 4.0, "b":80.5, "E_L":-70.6,
                        "g_L":300.0, 'E_ex':20.0, 'E_in': -85.0,
                        'tau_syn_ex':40.0, 'tau_syn_in': 20.0})

spike = nest.Create("spike_generator")
spike_times = [10.0, 400.0]
n_spikes = 2

# set spike times and heights
nest.SetStatus(spike, {"spike_times": spike_times})
delay = [1.0, 100.0]
weight = [0.1, -0.2]

conn_spec={"rule": "all_to_all"}
syn_spec_ex={'receptor_type': 0, 'weight': weight[0], 'delay': delay[0]}
syn_spec_in={'receptor_type': 0, 'weight': weight[1], 'delay': delay[1]}
nest.Connect(spike, neuron, conn_spec, syn_spec_ex)
nest.Connect(spike, neuron, conn_spec, syn_spec_in)
voltmeter = nest.Create('voltmeter')
nest.Connect(voltmeter, neuron)

#record = nest.CreateRecord("", ["V_m"], [neuron[0]], [0])

nest.Simulate(800.0)

dmm = nest.GetStatus(voltmeter)[0]
V_m = dmm["events"]["V_m"]
t = dmm["events"]["times"]
with open('test_aeif_cond_alpha_nest_non_mul.txt', 'w') as f:
    for i in range(len(t)):
        f.write("%s\t%s\n" % (t[i], V_m[i]))

#data_list = nest.GetRecordData(record)
#t=[row[0] for row in data_list]
#V_m=[row[1] for row in data_list]

#data = np.loadtxt('test_aeif_cond_alpha_nest.txt', delimiter="\t")
#t1=[x[0] for x in data ]
#V_m1=[x[1] for x in data ]
#print (len(t))
#print (len(t1))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(t, V_m, "r--")
plt.show()

#dV=[V_m[i*10+20]-V_m1[i] for i in range(len(t1))]
#print(dV)
#rmse =np.std(dV)/abs(np.mean(V_m))
#print("rmse : ", rmse, " tolerance: ", tolerance)
#if rmse>tolerance:
#    sys.exit(1)

#sys.exit(0)
