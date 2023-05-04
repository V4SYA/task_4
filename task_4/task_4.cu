/*Этот код решает уравнение теплопроводности методом простых итераций
на графическом процессоре (GPU) с помощью технологии CUDA.
Вначале определяются параметры модели, такие как точность, размер сетки и максимальное число итераций.
Затем выделяется память на хосте и девайсе для двух массивов A и Anew,
которые хранят значения температуры на сетке.*/

/*Основной цикл while выполняет итерации метода Якоби до тех пор,
   пока не будет достигнута максимальная допустимая ошибка или количество итераций не превысит заданный предел. 
   В каждой итерации происходит вызов функции cross_calc, которая вычисляет новые значения элементов 
   массива Anew на основе значений элементов массива A. После каждой 100-й итерации происходит 
   вызов функции get_error_matrix, которая вычисляет матрицу ошибок между массивами A и Anew, 
   а затем используется библиотека CUB для нахождения максимальной ошибки. Если максимальная ошибка 
   меньше заданного порога, то цикл while завершается.
*/

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <time.h>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BILLION 1000000000

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;

///////////////////////////////////////////////////
///Вычисление новых значений элементов 
///массива Anew на основе значений элементов исходного
///массива A и заданного шаблона вычислений
///////////////////////////////////////////////////
__global__
void cross_calc(double* A, double* Anew, size_t n){
    // Получаю индексы блоков и потоков
    
    size_t j = blockIdx.x;
    size_t i = threadIdx.x;

    if (i != 0 && j != 0){
        Anew[j * n + i] = 0.25 * (
            A[j * n + i - 1] + 
            A[j * n + i + 1] + 
            A[(j + 1) * n + i] + 
            A[(j - 1) * n + i]
        );
    
    }

}

///////////////////////////////////////////////////
///Вычисление ошибки между элементами двух массивов A и Anew
///////////////////////////////////////////////////
__global__
void get_error_matrix(double* A, double* Anew, double* out){
    // Получаю индекс
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Получаю макс. ошибку
    if (blockIdx.x != 0 && threadIdx.x != 0){
        out[idx] = std::abs(Anew[idx] - A[idx]);
    }
}


int main(int argc, char ** argv){

    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);

    int n, iter_max;
    double min_error;

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &iter_max);
    sscanf(argv[3], "%lf", &min_error);

    int full_size = n * n;
    double step = (CORNER_2 - CORNER_1) / (n - 1);
   
    // Инициализация массивов
    auto* A = new double[n * n];
    auto* Anew = new double[n * n];

    std::memset(A, 0, sizeof(double) * n * n);

    //Угловые значения
    A[0] = CORNER_1;
    A[n - 1] = CORNER_2;
    A[n * n - 1] = CORNER_3;
    A[n * (n - 1)] = CORNER_4;
   
    //Значения краёв сетки
    for (int i = 1; i < n - 1; i ++) {
        A[i] = CORNER_1 + i * step;
        A[n * i] = CORNER_1 + i * step;
        A[(n-1) + n * i] = CORNER_2 + i * step;
        A[n * (n-1) + i] = CORNER_4 + i * step;
    }

    std::memcpy(Anew, A, sizeof(double) * full_size);

    cudaSetDevice(3);

    double* dev_A, *dev_B, *dev_err, *dev_err_mat, *temp_stor = NULL;
    size_t tmp_stor_size = 0;

    int i = 0;
    double error = 1.0;

    nvtxRangePushA("Main loop");

    while (i < iter_max && error > min_error){
        i++;
        // Вычисление итерации
        cross_calc<<<n-1, n-1>>>(dev_A, dev_B, n);

        if (i % 100 == 0){
            // Получение ошибки
            get_error_matrix<<<n - 1, n - 1>>>(dev_A, dev_B, dev_err_mat);
            // Найти макс. ошибку
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            // Копирую память в хост
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }
       
        // Смена массивов
        std::swap(dev_A, dev_B);
    }

    nvtxRangePop();

    clock_gettime(CLOCK_REALTIME, &stop);
    double delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;

    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << i << std::endl;
    std::cout << "Time: " << delta << std::endl;

    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);

    delete[] A;
    delete[] Anew;
    return 0;
}
