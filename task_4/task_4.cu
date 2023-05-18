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

#include <iostream>
#include <cmath>
#include <cstring>
#include <time.h>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

char ERROR_WITH_ARGS[] = ">>> Not enough args\n";
char ERROR_WITH_ARG_1[] = ">>> Incorrect first param\n";
char ERROR_WITH_ARG_2[] = ">>> Incorrect second param\n";
char ERROR_WITH_ARG_3[] = ">>> Incorrect third param\n";

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
void cross_calc(double* A, double* Anew, size_t size){
    // get the block and thread indices
    
    size_t j = blockIdx.x;
    size_t i = threadIdx.x;
    // main cross computation. the average of 4 incident cells is taken
    if (i != 0 && j != 0){
       
        Anew[j * size + i] = 0.25 * (
            A[j * size + i - 1] + 
            A[j * size + i + 1] + 
            A[(j + 1) * size + i] + 
            A[(j - 1) * size + i]
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
    int max_iter, size;
    double min_error;

    // Проверка ввода данных
    if (argc < 4){
        std::cout << ERROR_WITH_ARGS << std::endl;
        exit(1);
    } else{
        size = atoi(argv[1]); // Размер сетки
        if (size == 0){
            std::cout << ERROR_WITH_ARG_1 << std::endl;
            exit(1);
        }
        max_iter = atoi(argv[2]); // Количество итераций
        if (max_iter == 0){
            std::cout << ERROR_WITH_ARG_2 << std::endl;
            exit(1);
        }
        min_error = atof(argv[3]); // Точность
        if (min_error == 0){
            std::cout << ERROR_WITH_ARG_3 << std::endl;
            exit(1);
        }
    }

    clock_t a = clock();

    int full_size = size * size;
    double step = (CORNER_2 - CORNER_1) / (size - 1);

    // Инициализация массивов
    auto* A = new double[size * size];
    auto* Anew = new double[size * size];

    std::memset(A, 0, sizeof(double) * size * size);

    // Угловые значения
    A[0] = CORNER_1;
    A[size - 1] = CORNER_2;
    A[size * size - 1] = CORNER_3;
    A[size * (size - 1)] = CORNER_4;

    // Значения краёв сетки
    for (int i = 1; i < size - 1; i ++) {
        A[i] = CORNER_1 + i * step;
        A[size * i] = CORNER_1 + i * step;
        A[(size-1) + size * i] = CORNER_2 + i * step;
        A[size * (size-1) + i] = CORNER_4 + i * step;
    }

    std::memcpy(Anew, A, sizeof(double) * full_size);
    
    // Выбор девайса
    cudaSetDevice(3);
    
    double* dev_A, *dev_B, *dev_err, *dev_err_mat, *temp_stor = NULL;
    size_t tmp_stor_size = 0;

    // Выделение памяти на 2 матрицы и переменная ошибки на устройстве 
    cudaError_t status_A = cudaMalloc(&dev_A, sizeof(double) * full_size);
    cudaError_t status_B = cudaMalloc(&dev_B, sizeof(double) * full_size);
    cudaError_t status = cudaMalloc(&dev_err, sizeof(double));

    // Некоторые действия по выделению памяти для выявления ошибок
    if (status != cudaSuccess){
        std::cout << "Device error variable allocation error " << status << std::endl;
        return status;
    }

    // Выделение памяти на устройстве для матрицы ошибок
    status = cudaMalloc(&dev_err_mat, sizeof(double) * full_size);
    if (status != cudaSuccess){
        std::cout << "Device error matrix allocation error " << status << std::endl;
        return status;
    }
    if (status_A != cudaSuccess){
        std::cout << "Kernel A allocation error " << status << std::endl;
        return status;
    } else if (status_B != cudaSuccess){
        std::cout << "Kernel B allocation error " << status << std::endl;
        return status;
    }

    status_A = cudaMemcpy(dev_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_A != cudaSuccess){
        std::cout << "Kernel A copy to device error " << status << std::endl;
        return status_A;
    }
    status_B = cudaMemcpy(dev_B, Anew, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_B != cudaSuccess){
        std::cout << "kernel B copy to device error " << status << std::endl;
        return status_B;
    }

    status = cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
    if (status != cudaSuccess){
        std::cout << "Max reduction error " << status << std::endl;
        return status;
    }

    status = cudaMalloc(&temp_stor, tmp_stor_size);
    if (status != cudaSuccess){
        std::cout << "Temporary storage allocation error " << status  << std::endl;
        return status;
    }

    int i = 0;
    double error = 1.0;

    nvtxRangePushA("Main loop");

    // Основной алгоритм
    while (i < max_iter && error > min_error){
        i++;
        // Вычисление итерации
        cross_calc<<<size-1, size-1>>>(dev_A, dev_B, size);

        if (i % 100 == 0){
            // Получение ошибки
            // кол-во потоков = (size-1)^2
            get_error_matrix<<<size - 1, size - 1>>>(dev_A, dev_B, dev_err_mat);
            
            // Находим максимальную ошибку
            // Результат в dev_err
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            
            // Копирую ошибку с устройства в память хоста
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }

        // Смена массивов
        std::swap(dev_A, dev_B);

    }

    nvtxRangePop();

    // Вывод массивов
    // cudaMemcpy(A, dev_A, sizeof(double) * full_size, cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Вывод результатов
    clock_t b = clock();
    double d = (double)(b-a)/CLOCKS_PER_SEC; // перевожу в секунды 
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << i << std::endl;
    std::cout << "Time: " << d << std::endl;

    // Очистка
    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);
    
    delete[] A;
    delete[] Anew;
    return 0;
}
