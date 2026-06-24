import nestgpu as ngpu
import numpy as np

neuron1 = ngpu.Create("aeif_cond_beta_multisynapse", 50000)
neuron2 = ngpu.Create("aeif_cond_beta_multisynapse", 50000)

ngpu.SetStatus(neuron1, {"V_m": {"distribution":"lognormal_clipped",
                                       "mu":10.0, "low":0.1,
                                       "high":100.0,
                                       "sigma":5.0}})
l = ngpu.GetStatus(neuron1, "V_m")

ngpu.Connect(neuron1, neuron2, {'rule': 'one_to_one'}, {'delay': {'distribution': 'lognormal_clipped', 'mu': 10.0, 'sigma': 5.0, 'low': 0.1, 'high': 100},
                                                        'weight': {'distribution': 'lognormal_clipped', 'mu': 10.0, 'sigma': 5.0, 'low': 0.1, 'high': 100}})

conn_list = ngpu.GetConnections(neuron1, neuron2)
conn_status = ngpu.GetConnectionStatus(conn_list)

delays = np.zeros(50000)
weights = np.zeros(50000)

for c, conns in enumerate(conn_status):
    delays[c] = conns['delay']
    weights[c] = conns['weight']

rng = np.random.default_rng()
baseline = rng.lognormal(mean=10.0, sigma=5.0, size=50000)
baseline = np.minimum(baseline, 100)
baseline = np.maximum(baseline, 0.1)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = ax[0, 0].hist(l, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, label="Vm")
n, bins, patches = ax[0, 1].hist(delays, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, label="Delays")
n, bins, patches = ax[1, 0].hist(weights, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, label="Weights")
n, bins, patches = ax[1, 1].hist(baseline, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, label="Baseline")


plt.grid(axis='y', alpha=0.75)
ax[0, 0].xlabel('Value')
ax[0, 1].xlabel('Value')
ax[1, 0].xlabel('Value')
ax[1, 1].xlabel('Value')

ax[0, 0].ylabel('Frequency')
ax[0, 1].ylabel('Frequency')
ax[1, 0].ylabel('Frequency')
ax[1, 1].ylabel('Frequency')
#plt.title('V_m Histogram')
#plt.text(23, 45, r'$\mu=15, b=3$')
plt.legend()
#maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig("lognormal_test.png")

#plt.show()
