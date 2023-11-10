/*
 *  diffusion_connection.cu
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





#include <config.h>
#include <stdio.h>
#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "diffusion_connection.h"
#include "syn_model.h"

using namespace diffusion_connection_ns;

int diffusion_connection::Init()
{
  type_ = i_diffusion_connection_model;
  n_param_ = N_PARAM;
  param_name_ = diffusion_connection_param_name;
  CUDAMALLOCCTRL("&d_param_arr_",&d_param_arr_, n_param_*sizeof(float));
  SetParam("drift_factor", 1.0);
  SetParam("diffusion_factor", 1.0);

  return 0;
}
