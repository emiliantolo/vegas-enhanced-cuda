#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "config.cuh"
#include "commons/common.cuh"

__global__
void setup_kernel(curandState *state, int n_cubes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n_cubes) { 
        curand_init(123, idx, 0, &state[idx]);
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *data, int tid) {
    if (blockSize >= 64) data[tid] += data[tid + 32];
    if (blockSize >= 32) data[tid] += data[tid + 16];
    if (blockSize >= 16) data[tid] += data[tid + 8];
    if (blockSize >= 8) data[tid] += data[tid + 4];
    if (blockSize >= 4) data[tid] += data[tid + 2];
    if (blockSize >= 2) data[tid] += data[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(double *input, double *output, int n) {
    extern __shared__ double data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + tid;
    unsigned int gridSize = blockSize * gridDim.x;
    data[tid] = 0;

    while (i < n) { data[tid] += input[i]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { data[tid] += data[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { data[tid] += data[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { data[tid] += data[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) warpReduce<blockSize>(data, tid);
    if (tid == 0) atomicAdd(output, data[0]);
}

template <int n_dim>
__global__ void vegasFill(int n_strat, int n_intervals, int n_edges, int n_cubes, double v_cubes, double beta, int *counts, double *weights, double *x_edges, double *dx_edges, double *dh, double *nh, double *sum_r, double *sum_s, curandState *dev_states) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < n_cubes) {

        //state
        curandState state = dev_states[idx];

        // get eval for every cube
        int neval = nh[idx];

        //reset weight strat
        double JF_l = 0;
        double JF2_l = 0;
        double Counts_l = 0;

        double y;
        double offset;
        int id[n_dim];
        double x[n_dim];

        for (int index = 0; index < neval; index++) {

            //get y
            double dy = 1.0 / n_strat;
            int tmp = idx;
            for (int i = 0; i < n_dim; i++) {
                double rand = curand_uniform(&state); //random number 0-1
                int q = tmp / n_strat;
                int r = tmp - (q * n_strat);
                tmp = q;
                y = rand * dy + r * dy;
                //get interval id, integer part of mapped point
                id[i] = (int) (y * n_intervals);
                //get interval offset, fractional part
                offset = (y * (double) n_intervals) - id[i];
                //get x
                x[i] = x_edges[i*n_edges+id[i]] + dx_edges[i*n_intervals+id[i]] * offset;
            }

            //eval
            double f = integrand(x, n_dim);

            //get jac
            double jac = 1.0;
            for (int i = 0; i < n_dim; i++) {
                jac *= n_intervals * dx_edges[i*n_intervals+id[i]];
            }

            double jf_t = f * jac;
            double jf2_t = jf_t * jf_t;

            //accumulate weight
            for (int i = 0; i < n_dim; i++) {
                atomicAdd(&(weights[i*n_intervals+id[i]]), jf2_t);
                atomicAdd(&(counts[i*n_intervals+id[i]]), 1);
            }

            //accumulate weight strat
            //accumulate += jf and jf2
            JF_l += jf_t;
            JF2_l += jf2_t;
            Counts_l += 1;

        }

        double Ih = JF_l / neval * v_cubes;
        double Sig2 = (JF2_l / neval * v_cubes * v_cubes) - (Ih * Ih);

        sum_r[idx] = Ih;
        sum_s[idx] = Sig2 / neval;

        dh[idx] = pow((1 - (1 / Counts_l)) * (v_cubes * v_cubes / Counts_l * JF2_l), beta);

        //state
        dev_states[idx] = state;
    
    }
}

__global__
void normalizeWeights(int n_dim, int n_intervals, int *counts, double *weights, double *d_sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        int dim = i / n_intervals;

        //normalize weight
        if (counts[i] != 0) {
            weights[i] /= counts[i];
        }

        atomicAdd(&d_sum[dim], weights[i]);
    }
}

__global__
void smoothWeights(int n_dim, int n_intervals, double alpha, double *weights, double *smoothed_weights, double *summed_weights, double *d_sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        int dim = i / n_intervals;
        int interval = i % n_intervals;

        double d_tmp;
        //smooth weight
        if (interval == 0) {
            d_tmp = (7.0 * weights[i] + weights[i+1]) / (8.0 * d_sum[dim]);
        } else if (interval == (n_intervals - 1)) {
            d_tmp = (weights[i-1] + 7.0 * weights[i]) / (8.0 * d_sum[dim]);
        } else {
            d_tmp = (weights[i-1] + 6.0 * weights[i] + weights[i+1]) / (8.0 * d_sum[dim]);
        }
        //smooth alpha
        if (d_tmp != 0) {
            d_tmp = pow((d_tmp - 1.0) / log(d_tmp), alpha);
        }
        smoothed_weights[i] = d_tmp;
        atomicAdd(&summed_weights[dim], d_tmp);
    }
}

__global__
void resetWeights(int n_dim, int n_intervals, int *counts, double *weights) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < (n_dim * n_intervals)) {

        //reset weights
        weights[i] = 0;
        counts[i] = 0;
    }
}

__global__
void updateNh(int n_cubes, int n_eval_it, double *dh, double *nh, double *dh_sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n_cubes) {
        
        // update nh
        int nh_s = dh[i] / *dh_sum * n_eval_it;
        nh[i] = nh_s < 2 ? 2 : nh_s;
    }
}


int main() {

    double startTime = cpuMilliSeconds();

    int n_eval_it = tot_eval / max_it;

    //vegas map
    int n_edges = n_intervals + 1;
    double *x_edges, *x_edges_d;
    double *dx_edges, *dx_edges_d;
    double *weights_d; //ndim,n_intervals
    int *counts_d; //ndim,n_intervals
    double *x_edges_old, *x_edges_old_d;
    double *dx_edges_old, *dx_edges_old_d;
    double *smoothed_weights, *smoothed_weights_d;
    double *summed_weights, *summed_weights_d;

    //vegas stratification
    int n_strat = (int) pow((n_eval_it / 2.0), 1.0 / n_dim);
    int n_cubes = pow(n_strat, n_dim);
    double v_cubes = pow((1.0 / n_strat), n_dim); 
    double *dh_d; //sample counts dampened
    double *nh, *nh_d; //statified sample counts per cube

    double *Results;
    double *Sigma2;

    double *res_s, *res_s_d;
    double *sig_s, *sig_s_d;
    double *dh_sum, *dh_sum_d;
    double *d_sum, *d_sum_d;

    double *sum_r_d;
    double *sum_s_d;

    curandState *dev_states;

    x_edges = (double*) malloc(n_dim*n_edges * sizeof(double));
    dx_edges = (double*) malloc(n_dim*n_intervals * sizeof(double));

    x_edges_old = (double*) malloc(n_dim*n_edges * sizeof(double));
    dx_edges_old = (double*) malloc(n_dim*n_intervals * sizeof(double));

    smoothed_weights = (double*) malloc(n_dim*n_intervals * sizeof(double));

    nh = (double*) malloc(n_cubes * sizeof(double));

    Results = (double*) malloc(max_it * sizeof(double));
    Sigma2 = (double*) malloc(max_it * sizeof(double));

    res_s = (double*) malloc(sizeof(double));
    sig_s = (double*) malloc(sizeof(double));
    dh_sum = (double*) malloc(sizeof(double));
    d_sum = (double*) malloc(n_dim*sizeof(double));
    summed_weights = (double*) malloc(n_dim*sizeof(double));

    checkCudaError(cudaMalloc(&x_edges_d, n_dim*n_edges*sizeof(double)));
    checkCudaError(cudaMalloc(&dx_edges_d, n_dim*n_intervals*sizeof(double)));
    checkCudaError(cudaMalloc(&x_edges_old_d, n_dim*n_edges*sizeof(double)));
    checkCudaError(cudaMalloc(&dx_edges_old_d, n_dim*n_intervals*sizeof(double)));
    checkCudaError(cudaMalloc(&weights_d, n_dim*n_intervals*sizeof(double)));
    checkCudaError(cudaMalloc(&counts_d, n_dim*n_intervals*sizeof(int)));
    checkCudaError(cudaMalloc(&smoothed_weights_d, n_dim*n_intervals*sizeof(double)));

    checkCudaError(cudaMalloc(&dh_d, n_cubes*sizeof(double)));
    checkCudaError(cudaMalloc(&nh_d, n_cubes*sizeof(double)));

    checkCudaError(cudaMalloc(&dev_states, n_cubes*sizeof(curandState)));

    checkCudaError(cudaMalloc(&res_s_d, sizeof(double)));
    checkCudaError(cudaMalloc(&sig_s_d, sizeof(double)));
    checkCudaError(cudaMalloc(&dh_sum_d, sizeof(double)));

    checkCudaError(cudaMalloc(&d_sum_d, n_dim*sizeof(double)));
    checkCudaError(cudaMalloc(&summed_weights_d, n_dim*sizeof(double)));

    checkCudaError(cudaMalloc(&sum_r_d, n_cubes*sizeof(double)));
    checkCudaError(cudaMalloc(&sum_s_d, n_cubes*sizeof(double)));

    // cuda parameters
    int gridSizeRed = n_cubes / blockSizeRed + (int) ((n_cubes % blockSizeRed) != 0);
    int gridSizeUpd = n_cubes / blockSizeUpd + (int) ((n_cubes % blockSizeUpd) != 0);
    int gridSizeFill = n_cubes / blockSizeFill + (int) ((n_cubes % blockSizeFill) != 0);
    int gridSizeIntervals = n_dim * n_intervals / blockSizeIntervals + (int) (((n_dim * n_intervals) % blockSizeIntervals) != 0);    
    int gridSizeInit = n_cubes / blockSizeInit + (int) ((n_cubes % blockSizeInit) != 0);

    printf("Strat: %i\n", n_strat);

    printf("Cubes: %i\n", n_cubes);

    // init dx_edges and x_edges
    double step = 1.0 / n_intervals;
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_edges; j++) {
            x_edges[i*n_edges+j] = j * step;
        }
        for (int j = 0; j < n_intervals; j++) {
            dx_edges[i*n_intervals+j] = x_edges[i*n_edges+j+1] - x_edges[i*n_edges+j];
        }
    }

    // init nh
    for (int i = 0; i < n_cubes; i++) {
        int nh_s = 1.0 / n_cubes * n_eval_it;
        nh[i] = nh_s < 2 ? 2 : nh_s;
    }

    // init d_sum and summed_weights
    for (int i = 0; i < n_dim; i++) {
        d_sum[i] = 0.0;
        summed_weights[i] = 0.0;
    }

    checkCudaError(cudaMemcpy(x_edges_d, x_edges, n_dim*n_edges*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dx_edges_d, dx_edges, n_dim*n_intervals*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(nh_d, nh, n_cubes*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_sum_d, d_sum, n_dim*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(summed_weights_d, summed_weights, n_dim*sizeof(double), cudaMemcpyHostToDevice));

    //init rng
    setup_kernel<<<gridSizeInit,blockSizeInit>>>(dev_states, n_cubes);

    checkCudaError(cudaPeekAtLastError());
    checkCudaError(cudaDeviceSynchronize());

    double res = 0;
    double sigmas = 0;

    // get the range of stream priorities for this device
    int priority_high, priority_low;
    checkCudaError(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    // create streams with highest and lowest available priorities
    cudaStream_t st_low;
    checkCudaError(cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_high));

    int it = 0;

    double startTimeIt = cpuMilliSeconds();

    do {

        //reset res and sig
        *res_s = 0;
        *sig_s = 0;
        *dh_sum = 0.0;

        checkCudaError(cudaMemcpy(res_s_d, res_s, sizeof(double), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(sig_s_d, sig_s, sizeof(double), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(dh_sum_d, dh_sum, sizeof(double), cudaMemcpyHostToDevice));


        //call kernel
        vegasFill<n_dim><<<gridSizeFill,blockSizeFill>>>(n_strat, n_intervals, n_edges, n_cubes, v_cubes, beta, counts_d, weights_d, x_edges_d, dx_edges_d, dh_d, nh_d, sum_r_d, sum_s_d, dev_states);

        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());

        normalizeWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_low>>>(n_dim, n_intervals, counts_d, weights_d, d_sum_d);
        smoothWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_low>>>(n_dim, n_intervals, alpha, weights_d, smoothed_weights_d, summed_weights_d, d_sum_d);

        reduce<blockSizeRed><<<gridSizeRed,blockSizeRed,sizeof(double)*blockSizeRed>>>(dh_d, dh_sum_d, n_cubes);
        updateNh<<<gridSizeUpd,blockSizeUpd>>>(n_cubes, n_eval_it, dh_d, nh_d, dh_sum_d);

        if (it >= skip) {
            reduce<blockSizeRed><<<gridSizeRed,blockSizeRed,sizeof(double)*blockSizeRed>>>(sum_r_d, res_s_d, n_cubes);
            reduce<blockSizeRed><<<gridSizeRed,blockSizeRed,sizeof(double)*blockSizeRed>>>(sum_s_d, sig_s_d, n_cubes);
        }

        checkCudaError(cudaMemcpyAsync(x_edges, x_edges_d, n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToHost, st_low));
        checkCudaError(cudaMemcpyAsync(dx_edges, dx_edges_d, n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToHost, st_low));
        checkCudaError(cudaMemcpyAsync(x_edges_old, x_edges_d, n_dim*n_edges*sizeof(double), cudaMemcpyDeviceToHost, st_low));
        checkCudaError(cudaMemcpyAsync(dx_edges_old, dx_edges_d, n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToHost, st_low));
        checkCudaError(cudaMemcpyAsync(summed_weights, summed_weights_d, n_dim*sizeof(double), cudaMemcpyDeviceToHost, st_low));
        checkCudaError(cudaMemcpyAsync(smoothed_weights, smoothed_weights_d, n_dim*n_intervals*sizeof(double), cudaMemcpyDeviceToHost, st_low));

        for (int i = 0; i < n_dim; i++) {

            double delta_weights = summed_weights[i] / n_intervals;

            // update map
            int old_interval = 0;
            int new_interval = 1;
            double acc = 0;

            while (true) {
                acc += delta_weights;
                while (acc > smoothed_weights[i*n_intervals+old_interval]) {
                    acc -= smoothed_weights[i*n_intervals+old_interval];
                    old_interval++;
                }
                x_edges[i*n_edges+new_interval] = x_edges_old[i*n_edges+old_interval] + acc / smoothed_weights[i*n_intervals+old_interval] * dx_edges_old[i*n_intervals+old_interval];
                dx_edges[i*n_intervals+new_interval-1] = x_edges[i*n_edges+new_interval] - x_edges[i*n_edges+new_interval-1];
                new_interval++;
                if (new_interval >= n_intervals) {
                    break;
                } 
            }
            dx_edges[i*n_intervals+n_intervals-1] = x_edges[i*n_edges+n_edges-1] - x_edges[i*n_edges+n_edges-2];

            //reset d_sum and summed_weights
            d_sum[i] = 0.0;
            summed_weights[i] = 0.0;
        }

        checkCudaError(cudaMemcpyAsync(x_edges_d, x_edges, n_dim*n_edges*sizeof(double), cudaMemcpyHostToDevice, st_low));
        checkCudaError(cudaMemcpyAsync(dx_edges_d, dx_edges, n_dim*n_intervals*sizeof(double), cudaMemcpyHostToDevice, st_low));
        checkCudaError(cudaMemcpyAsync(d_sum_d, d_sum, n_dim*sizeof(double), cudaMemcpyHostToDevice, st_low));
        checkCudaError(cudaMemcpyAsync(summed_weights_d, summed_weights, n_dim*sizeof(double), cudaMemcpyHostToDevice, st_low));

        resetWeights<<<gridSizeIntervals,blockSizeIntervals,0,st_low>>>(n_dim, n_intervals, counts_d, weights_d);

        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());

        it++;

        if (it > skip) {
            checkCudaError(cudaMemcpy(res_s, res_s_d, sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaError(cudaMemcpy(sig_s, sig_s_d, sizeof(double), cudaMemcpyDeviceToHost));

            Results[it - skip - 1] = *res_s;
            Sigma2[it - skip - 1] = *sig_s;

            //results
            res = 0;
            sigmas = 0;
            for (int i = 0; i < it - skip; i++) {
                res += Results[i] / Sigma2[i];
                sigmas += 1.0 / Sigma2[i];
            }
            res /= sigmas;
        }

    } while(it < max_it);

    double elapsedTimeIt = cpuMilliSeconds() - startTimeIt;

    //memory
    free(x_edges);
    free(dx_edges);
    free(nh);
    free(Results);
    free(Sigma2);
    free(res_s);
    free(sig_s);
    free(dh_sum);
    free(d_sum);
    free(summed_weights);

    checkCudaError(cudaFree(x_edges_d));
    checkCudaError(cudaFree(dx_edges_d));
    checkCudaError(cudaFree(x_edges_old_d));
    checkCudaError(cudaFree(dx_edges_old_d));
    checkCudaError(cudaFree(weights_d));
    checkCudaError(cudaFree(counts_d));
    checkCudaError(cudaFree(smoothed_weights_d));
    checkCudaError(cudaFree(dh_d));
    checkCudaError(cudaFree(nh_d));
    checkCudaError(cudaFree(dev_states));
    checkCudaError(cudaFree(res_s_d));
    checkCudaError(cudaFree(sig_s_d));
    checkCudaError(cudaFree(dh_sum_d));
    checkCudaError(cudaFree(d_sum_d));
    checkCudaError(cudaFree(summed_weights_d));
    checkCudaError(cudaFree(sum_r_d));
    checkCudaError(cudaFree(sum_s_d));

    double elapsedTime = cpuMilliSeconds() - startTime;

    printf("Result: %.8f\n", res);
    printf("Error: %.8f\n", 1.0 / sqrt(sigmas));
    printf("Time elapsed %f ms\n", elapsedTime);
    printf("Iteration avg time %f ms\n", elapsedTimeIt / max_it);

    return(0);
}
