/*
 *  remote_spike.h
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





#ifndef REMOTE_SPIKE_H
#define REMOTE_SPIKE_H

extern __constant__ bool have_remote_spike_height;

__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id,
                                    float *spike_height);

__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id);

extern __device__ int NExternalTargetHost;
extern __device__ int MaxSpikePerHost;

extern int *d_ExternalSpikeNum;
extern __device__ int *ExternalSpikeNum;

extern int *d_ExternalSpikeSourceNode; // [MaxSpikeNum];
extern __device__ int *ExternalSpikeSourceNode;

extern float *d_ExternalSpikeHeight; // [MaxSpikeNum];
extern __device__ float *ExternalSpikeHeight;

extern int *d_ExternalTargetSpikeNum;
extern __device__ int *ExternalTargetSpikeNum;

extern int *d_ExternalTargetSpikeNodeId;
extern __device__ int *ExternalTargetSpikeNodeId;

extern float *d_ExternalTargetSpikeHeight;
extern __device__ float *ExternalTargetSpikeHeight;

//extern int *d_NExternalNodeTargetHost;
extern __device__ int *NExternalNodeTargetHost;

//extern int **d_ExternalNodeTargetHostId;
extern __device__ int **ExternalNodeTargetHostId;

//extern int **d_ExternalNodeId;
extern __device__ int **ExternalNodeId;

//extern int *d_ExternalSourceSpikeNum;
//extern __device__ int *ExternalSourceSpikeNum;

extern int *d_ExternalSourceSpikeNodeId;
extern __device__ int *ExternalSourceSpikeNodeId;

extern float *d_ExternalSourceSpikeHeight;
extern __device__ float *ExternalSourceSpikeHeight;

extern int *d_ExternalTargetSpikeIdx0;
extern __device__ int *ExternalTargetSpikeIdx0;

extern int *d_ExternalSourceSpikeIdx0;

extern int *h_ExternalTargetSpikeNum;
extern int *h_ExternalTargetSpikeIdx0;
extern int *h_ExternalTargetSpikeNodeId;

extern int *h_ExternalSourceSpikeNum;
extern int *h_ExternalSourceSpikeIdx0;
extern int *h_ExternalSourceSpikeNodeId;

//extern int *h_ExternalSpikeNodeId;

extern float *h_ExternalSpikeHeight;

__device__ void PushExternalSpike(int i_source, float height);

__device__ void PushExternalSpike(int i_source);

__global__ void countExternalSpikesPerTargetHost();

__global__ void organizeExternalSpikesPerTargetHost();

__global__ void DeviceExternalSpikeInit(int n_hosts,
					int max_spike_per_host,
		      			int *ext_spike_num,
					int *ext_spike_source_node,
                                        float *ext_spike_height,
					int *ext_target_spike_num,
					int *ext_target_spike_idx0,
					int *ext_target_spike_node_id,
                                        float *ext_target_spike_height,
					int *n_ext_node_target_host,
					int **ext_node_target_host_id,
					int **ext_node_id
					);

__global__ void DeviceExternalSpikeInit(int n_hosts,
					int max_spike_per_host,
		      			int *ext_spike_num,
					int *ext_spike_source_node,
					int *ext_target_spike_num,
					int *ext_target_spike_idx0,
					int *ext_target_spike_node_id,
					int *n_ext_node_target_host,
					int **ext_node_target_host_id,
					int **ext_node_id
					);


#endif
