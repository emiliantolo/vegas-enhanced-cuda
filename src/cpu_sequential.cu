#include <stdio.h>
#include <random>
#include "config.cuh"
#include "commons/common.cuh"

int main() {

    double startTime = cpuMilliSeconds();

    int n_eval_it = tot_eval / max_it;

    //vegas map
    int n_edges = n_intervals + 1;
    double *x_edges;
    double *dx_edges;
    double *weights;
    int *counts;
    double *x_edges_old;
    double *dx_edges_old;
    double *smoothed_weights;

    //vegas stratification
    int n_strat = (int) pow((n_eval_it / 2.0), 1.0 / n_dim);
    int n_cubes = pow(n_strat, n_dim);
    double v_cubes = pow((1.0 / n_strat), n_dim); 
    double *dh;
    double *nh;

    double *Results;
    double *Sigma2;

    double res_s;
    double sig_s;

    double y;
    double offset;
    int *id;
    double *x;

    x_edges = new double [n_dim*n_edges];
    dx_edges = new double [n_dim*n_intervals];

    dh = new double [n_cubes];
    nh = new double [n_cubes];

    Results = new double [max_it];
    Sigma2 = new double [max_it];

    x_edges = new double [n_dim*n_edges];
    dx_edges = new double [n_dim*n_intervals];
    x_edges_old = new double [n_dim*n_edges];
    dx_edges_old = new double [n_dim*n_intervals];
    weights = new double [n_dim*n_intervals];
    counts = new int [n_dim*n_intervals];
    smoothed_weights = new double [n_dim*n_intervals];

    dh = new double [n_cubes];
    nh = new double [n_cubes];
    
    id = new int [n_dim];
    x = new double [n_dim];

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

    //init rng
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    double res = 0;
    double sigmas = 0;

    int it = 0;

    double startTimeIt = cpuMilliSeconds();

    do {

        res_s = 0;
        sig_s = 0;

        //vegas fill
        for (int idx = 0; idx < n_cubes; idx++) {

            // get eval for every cube
            int neval = nh[idx];

            //reset weight strat
            double JF_l = 0;
            double JF2_l = 0;
            double Counts_l = 0;

            for (int index = 0; index < neval; index++) {

                //get y
                double dy = 1.0 / n_strat;
                int tmp = idx;
                for (int i = 0; i < n_dim; i++) {
                    double rand = distribution(generator); //random number 0-1
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

                double jac = 1.0;
                for (int i = 0; i < n_dim; i++) {
                    jac *= n_intervals * dx_edges[i*n_intervals+id[i]];
                }

                double jf_t = f * jac;
                double jf2_t = jf_t * jf_t;

                //accumulate weight
                for (int i = 0; i < n_dim; i++) {
                    weights[i*n_intervals+id[i]] += jf2_t;
                    counts[i*n_intervals+id[i]] += 1;
                }

                //accumulate weight strat
                //accumulate += jf and jf2
                JF_l += jf_t;
                JF2_l += jf2_t;
                Counts_l += 1;

            }

            double Ih = JF_l / neval * v_cubes;
            double Sig2 = (JF2_l / neval * v_cubes * v_cubes) - (Ih * Ih);

            if (it >= skip) {
                res_s += Ih;
                sig_s += Sig2 / neval;
            }

            dh[idx] = pow((1 - (1 / Counts_l)) * (v_cubes * v_cubes / Counts_l * JF2_l), beta);
        }

        //update all
        for (int i = 0; i < n_dim; i++) {

            //smooth weight
            for (int j = 0; j < n_intervals; j++) {
                if (counts[i*n_intervals+j] != 0) {
                    weights[i*n_intervals+j] /= counts[i*n_intervals+j];
                }
            }
            
            double d_tmp;
            double d_sum = 0.0;
            for (int j = 0; j < n_intervals; j++) {
                d_sum += weights[i*n_intervals+j];
            }

            double summed_weights = 0;
            for (int j = 0; j < n_intervals; j++) {
                if (j == 0) {
                    d_tmp = (7.0 * weights[i*n_intervals+0] + weights[i*n_intervals+1]) / (8.0 * d_sum);
                } else if (j == (n_intervals - 1)) {
                    d_tmp = (weights[i*n_intervals+n_intervals-2] + 7.0 * weights[i*n_intervals+n_intervals-1]) / (8.0 * d_sum);
                } else {
                    d_tmp = (weights[i*n_intervals+j-1] + 6.0 * weights[i*n_intervals+j] + weights[i*n_intervals+j+1]) / (8.0 * d_sum);
                }
                //smooth alpha
                if (d_tmp != 0) {
                    d_tmp = pow((d_tmp - 1.0) / log(d_tmp), alpha);
                }
                smoothed_weights[i*n_intervals+j] = d_tmp;
                summed_weights += d_tmp;
            }
            double delta_weights = summed_weights / n_intervals;

            // update map
            for (int j = 0; j < n_edges; j++) {
                x_edges_old[i*n_edges+j] = x_edges[i*n_edges+j];
            }
            for (int j = 0; j < n_intervals; j++) {
                dx_edges_old[i*n_intervals+j] = dx_edges[i*n_intervals+j];
            }

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

            //reset weights
            for (int j = 0; j < n_intervals; j++) {
                weights[i*n_intervals+j] = 0;
                counts[i*n_intervals+j] = 0;
            }

        }

        //sum dh
        double dh_sum = 0;
        for (int i = 0; i < n_cubes; i++) {
            
            // sum dh
            dh_sum += dh[i];
        }

        //update nh
        for (int i = 0; i < n_cubes; i++) {

            // update nh
            int nh_s = dh[i] / dh_sum * n_eval_it;
            nh[i] = nh_s < 2 ? 2 : nh_s;
        }

        it++;

        if (it > skip) {

            Results[it - skip - 1] = res_s;
            Sigma2[it - skip - 1] = sig_s;

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

    double elapsedTimeIt = cpuMilliSeconds() - startTime;

    delete(dh);
    delete(nh);
    delete(Results);
    delete(Sigma2);

    delete(x_edges);
    delete(dx_edges);
    delete(x_edges_old);
    delete(dx_edges_old);
    delete(weights);
    delete(counts);
    delete(smoothed_weights);
    
    delete(id);
    delete(x);

    double elapsedTime = cpuMilliSeconds() - startTime;

    printf("Result: %.8f\n", res);
    printf("Error: %.8f\n", 1.0 / sqrt(sigmas));
    printf("Time elapsed %f ms\n", elapsedTime);
    printf("Iteration avg time %f ms\n", elapsedTimeIt / max_it);
    
    return(0);
}

