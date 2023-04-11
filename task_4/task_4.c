#include <cstdlib>
#include <cstdio>
#include <malloc.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
__global__ void heat(const double* A, double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size + 1 && j > 0 && j < size + 1) {
        Anew[i * (size + 2) + j] = 0.25 * (A[(i + 1) * (size + 2) + j]
            + A[(i - 1) * (size + 2) + j]
            + A[i * (size + 2) + j - 1]
            + A[i * (size + 2) + j + 1]);
    }
}
__global__ void heatError(const double* A, double* Anew, int size, double error, double* er_1d) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size + 1 && j > 0 && j < size + 1) {
        Anew[i * (size + 2) + j] = 0.25 * (A[(i + 1) * (size + 2) + j]
            + A[(i - 1) * (size + 2) + j]
            + A[i * (size + 2) + j - 1]
            + A[i * (size + 2) + j + 1]);
        int idx_1d = (j * i) - 1;
        er_1d[idx_1d] = max(error, Anew[i * (size + 2) + j] - A[i * (size + 2) + j]);
    }
}
__global__ void errorReduce(double* er_1d, double* er_blocks, int size) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int gsz = blockDim.x * gridDim.x;

    double error = er_1d[0];

    for (int i = gid; i < size; i += gsz)
        error = max(error, er_1d[i]);

    extern __shared__ double shArr[];

    shArr[tid] = error;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz /= 2) {
        if (tid < sz)
            shArr[tid] = max(shArr[tid + sz], shArr[tid]);
        __syncthreads();
    }
    if (tid == 0)
        er_blocks[blockIdx.x] = shArr[0];
}

int main(int argc, char* argv[]) {
    int size, iter_max;
    double tol;
    
    tol = strtod(argv[1], NULL);
    size = atoi(argv[2]);
    iter_max = atoi(argv[3]);

    double* A = (double*)malloc((size + 2) * (size + 2) * sizeof(double));
    double* Anew = (double*)malloc((size + 2) * (size + 2) * sizeof(double));

    dim3 BS(32, 32, 1);
    dim3 GS((size + 2 + 31) / 32, (size + 2 + 31) / 32, 1);

    double* d_A, * d_Anew;

    cudaMalloc((void**)&d_A, sizeof(double) * (size + 2) * (size + 2));
    cudaMalloc((void**)&d_Anew, sizeof(double) * (size + 2) * (size + 2));

    int iter_host = 0;
    double error_host = 1.0;
    double add_grad_host = 10.0 / (size + 2);

    int len_host = size + 2;
    for (int i = 0; i < size + 2; i++)
    {
        A[i * len_host] = 10 + add_grad_host * i;
        A[i] = 10 + add_grad_host * i;
        A[len_host * (size + 1) + i] = 20 + add_grad_host * i;
        A[len_host * i + size + 1] = 20 + add_grad_host * i;

        Anew[len_host * i] = A[i * len_host];
        Anew[i] = A[i];
        Anew[len_host * (size + 1) + i] = A[len_host * (size + 1) + i];
        Anew[len_host * i + size + 1] = A[len_host * i + size + 1];
    }


    cudaMemcpy(d_A, A, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);

    double* d_err_1d;

    cudaMalloc(&d_err_1d, sizeof(double) * (size * size));

    dim3 errBS(1024, 1, 1);
    dim3 errGS(ceil((size * size) / (float)errBS.x), 1, 1);

    double* dev_out;
    cudaMalloc(&dev_out, sizeof(double) * errGS.x);


    double* d_ptr;
    cudaMalloc((void**)(&d_ptr), sizeof(double));

    cudaDeviceSynchronize();
    while ((error_host > tol) && (iter_host < iter_max)) {
        iter_host++;

        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            error_host = 0.0;
            heatError << <GS, BS >> > (d_A, d_Anew, size, error_host, d_err_1d);
            errorReduce << <errGS, errBS, (errBS.x) * sizeof(double) >> > (d_err_1d, dev_out, size * size);
            errorReduce << <1, errBS, (errBS.x) * sizeof(double) >> > (dev_out, d_err_1d, errGS.x);
            cudaMemcpy(&error_host, &d_err_1d[0], sizeof(double), cudaMemcpyDeviceToHost);
        }

        else heat << <GS, BS >> > (d_A, d_Anew, size);

        d_ptr = d_A;
        d_A = d_Anew;
        d_Anew = d_ptr;

        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            printf("%d : %lf\n", iter_host, error_host);
            fflush(stdout);
        }
    }

    printf("%d : %lf\n", iter_host, error_host);
    return 0;
}