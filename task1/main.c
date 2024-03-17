//
// Created by pavlusha on 29.02.24.
//

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

const int NUM_THREADS[] = {1, 4, 8, 12};
const int NET_SIZE[] = {100, 300, 500, 1000};
const int BLOCK_SIZES[] = {1, 8, 16, 32, 64};

const double EPS = 0.01;

typedef struct net {
    double h;
    int NS;
    int BS;

    double **f;
    double **u;
} net_t;

typedef double (*fun_xy)(double, double);

double d_f(double x, double y) {
    return 0;
}

double input_f(double x, double y) {
    return 52;
}

double **create_double_2d_arr(size_t sz) {
    double **arr = calloc(sz, sizeof(*arr));
    for (int i = 0; i < sz; i++)
        arr[i] = calloc(sz, sizeof(*arr[i]));
    return arr;
}

void free_double_2d_arr(double **arr, size_t sz) {
    for (int i = 0; i < sz; i++)
        free(arr[i]);
    return free(arr);
}

static double min(double a, double b) { return a < b ? a : b; }

static double max(double a, double b) { return a > b ? a : b; }

net_t *create_net(int n, fun_xy g, fun_xy f, int BS) {
    net_t *my_net = malloc(sizeof(*my_net));
    my_net->NS = n;
    my_net->h = 1.0 / (n + 1);
    my_net->u = create_double_2d_arr(n + 2);
    my_net->f = create_double_2d_arr(n + 2);
    my_net->BS = BS;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            my_net->f[i][j] = f(i * my_net->h, j * my_net->h);
        }
    }

    for (int i = 0; i <= n + 1; i++) {
        my_net->u[i][0] = g(i * my_net->h, 0);
        my_net->u[i][n + 1] = g(i * my_net->h, (n + 1) * my_net->h);
        my_net->u[0][i] = g(0, i * my_net->h);
        my_net->u[n + 1][i] = g((n + 1) * my_net->h, i * my_net->h);
    }
    return my_net;
}

double approximate_block(net_t *my_net, int i, int j) {
    int li = 1 + i * my_net->BS;
    int lj = 1 + j * my_net->BS;

    double dmax = 0;
    for (int ii = li; ii <= min(li + my_net->BS - 1, my_net->NS); ii++) {
        for (int jj = lj; jj <= min(lj + my_net->BS - 1, my_net->NS); jj++) {
            double temp = my_net->u[ii][jj];
            my_net->u[ii][jj] = 0.25 * (my_net->u[ii - 1][jj] + my_net->u[ii][jj - 1] + my_net->u[ii + 1][jj] +
                                        my_net->u[ii][jj + 1] - my_net->h * my_net->h * my_net->f[ii][jj]);
            double dm = fabs(temp - my_net->u[ii][jj]);
            dmax = max(dmax, dm);
        }
    }

    return dmax;
}

size_t approximate(net_t *my_net) {
    size_t iter = 0;
    int num_block = my_net->NS / my_net->BS + (my_net->NS % my_net->BS != 0);
    double dmax = 0;
    double *dm = calloc(num_block, sizeof(*dm));

    do {
        iter++;
        dmax = 0;
        for (int nx = 0; nx < num_block; nx++) {
            dm[nx] = 0;

            int i, j;
            double d;
#pragma omp parallel for shared(nx, dm) private(i, j, d)
            for (i = 0; i <= nx; i++) {
                j = nx - i;
                d = approximate_block(my_net, i, j);
                dm[i] = max(d, dm[i]);
            }
        }

        for (int nx = num_block - 1; nx >= 1; nx--) {
            int i, j;
            double d;
#pragma omp parallel for shared(nx, dm) private(i, j, d)
            for (i = num_block - nx; i < num_block; i++) {
                j = 2 * (num_block - 1) - nx - i + 1;
                d = approximate_block(my_net, i, j);
                dm[i] = max(d, dm[i]);
            }
        }

        for (int i = 0; i < num_block; i++) {
            dmax = max(dmax, dm[i]);
        }

    } while (dmax > EPS);
    free(dm);
    return iter;
}

FILE *create_resulting_file(char *name) {
    FILE *file = fopen(name, "w");

    for (int i = 0; i < sizeof(NUM_THREADS) / sizeof(NUM_THREADS[0]); ++i) {
        fprintf(file, "%d ", NUM_THREADS[i]);
    }
    fprintf(file, "\n");

    for (int i = 0; i < sizeof(NET_SIZE) / sizeof(NET_SIZE[0]); ++i) {
        fprintf(file, "%d ", NET_SIZE[i]);
    }

    fprintf(file, "\n");

    for (int i = 0; i < sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]); ++i) {
        fprintf(file, "%d ", BLOCK_SIZES[i]);
    }
    fprintf(file, "\n");
    return file;
}

int main() {
    FILE *file = create_resulting_file("result_function_1_test.txt");
    for (int i = 0; i < sizeof(NUM_THREADS) / sizeof(NUM_THREADS[0]); ++i) {
        for (int d = 0; d < sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]); ++d) {
            fprintf(file, "| $blockSize = %d$ |", BLOCK_SIZES[d]);
            for (int l = 0; l < sizeof(NET_SIZE) / sizeof(NET_SIZE[0]); ++l) {
                omp_set_num_threads(NUM_THREADS[i]);
                net_t *my_net = create_net(NET_SIZE[l], input_f, d_f, BLOCK_SIZES[d]);
                double t_start = omp_get_wtime();
                size_t count_iter = approximate(my_net);

                double t_end = omp_get_wtime();

                double max_e = -INFINITY;
                double min_e = INFINITY;
                double h = my_net->h;
                printf("################# RESULT FOR %d THREADS, %d BS, %d NS  #################\n", NUM_THREADS[i],
                       BLOCK_SIZES[d], NET_SIZE[l]);
                printf("TIME: %.6f\n", t_end - t_start);
                fprintf(file, " $%f$ |", t_end - t_start);
//                printf("COUNT OF ITERATION: %zu\n", count_iter);
//                for (int i = 1; i <= NET_SIZE[l]; i++) {
//                    for (int j = 1; j <= NET_SIZE[l]; j++) {
//                        max_e = max(max_e, fabs(input_f(i * h, j * h) - my_net->u[i][j]));
//                        min_e = min(min_e, fabs(input_f(i * h, j * h) - my_net->u[i][j]));
//                    }
//                }
//                printf("AVERAGE ERROR: %f\n", max_e / (NET_SIZE[l] * NET_SIZE[l]));
//                printf("MAX ERROR: %f\n", max_e);
//                printf("MIN ERROR: %f\n", min_e);

                free_double_2d_arr(my_net->u, my_net->NS + 2);
                free_double_2d_arr(my_net->f, my_net->NS + 2);
                free(my_net);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return 0;
}