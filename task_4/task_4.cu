/*Этот код решает уравнение теплопроводности методом простых итераций
на графическом процессоре (GPU) с помощью технологии CUDA.
Вначале определяются параметры модели, такие как точность, размер сетки и максимальное число итераций.
Затем выделяется память на хосте и девайсе для двух массивов A и Anew,
которые хранят значения температуры на сетке.*/

/*В основном цикле while выполняются итерации решения уравнения до тех пор,
пока ошибка не станет меньше заданной точности или число итераций не достигнет максимального значения.
    * Увеличивается счетчик итераций iter_host.
    * Если iter_host кратно 150 или равно 1, то вычисляется ошибка error_host, используя функцию heatError,
         и уменьшается размер ошибки, используя функцию errorReduce.
    * Затем текущее значение ошибки error_host копируется с устройства на хост.
    * Если iter_host кратно 150 или равно 1, то текущее значение ошибки выводится на экран.
    * Выполняется функция heat для обновления значений матрицы d_Anew.
    * Указатели d_A и d_Anew меняются местами.
    * Цикл повторяется до тех пор, пока ошибка error_host не станет меньше заданного порогового значения tol
         или пока не будет достигнуто максимальное число итераций iter_max.
*/

#include <cstdlib>
#include <cstdio>
#include <malloc.h>

////////////////////////////////////////////////////////////////////////////////
//Определяет индексы элементов матрицы, для которых нужно выполнить обновление,
//и вычисляет новые значения элементов на основе значений соседних элементов.
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
//Определяет индексы элементов матрицы, для которых нужно выполнить обновление,
//и вычисляет новые значения элементов на основе значений соседних элементов. 
//Также функция вычисляет ошибку между новым и старым значением элемента и сохраняет ее в одномерном массиве er_1d
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
//Вычисление максимальной ошибки в массиве er_1d и сохранение ее в массиве er_blocks
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// Основная точка входа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    int size, iter_max;
    double tol;
    
    tol = strtod(argv[1], NULL);
    size = atoi(argv[2]);
    iter_max = atoi(argv[3]);

    double* A = (double*)malloc((size + 2) * (size + 2) * sizeof(double));
    double* Anew = (double*)malloc((size + 2) * (size + 2) * sizeof(double));

   //Объявление трёхмерного блока потоков с размерами 32x32x1
    dim3 BS(32, 32, 1);
    //Объявление трёхмерной сетки блоков потоков с размерами, 
    //вычисленными по формуле ((size + 2 + 31) / 32) на каждую ось
    dim3 GS((size + 2 + 31) / 32, (size + 2 + 31) / 32, 1);

    double* d_A, * d_Anew;

    //Выделение памяти на устройстве для массива A размером (size + 2) * (size + 2) 
    //элементов типа double и привязка указателя d_A к началу выделенной области памяти
    cudaMalloc((void**)&d_A, sizeof(double) * (size + 2) * (size + 2));
    //Выделение памяти на устройстве для массива Anew размером (size + 2) * (size + 2)
    //элементов типа double и привязка указателя d_Anew к началу выделенной области памяти
    cudaMalloc((void**)&d_Anew, sizeof(double) * (size + 2) * (size + 2));

    int iter_host = 0;
    double error_host = 1.0;
    double add_grad_host = 10.0 / (size + 2);

    int len_host = size + 2;
    //Инициализируем массивы A и Anew, а также заполняем граничные значения
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

    //Копирование данных из массива A в память на устройстве, указатель на которую хранится в d_A
    cudaMemcpy(d_A, A, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);
    //Копирование данных из массива Anew в память на устройстве, указатель на которую хранится в d_Anew
    cudaMemcpy(d_Anew, Anew, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);

    double* d_err_1d;
    
    //Выделение памяти на устройстве для массива ошибок размером size * size элементов типа double
    //и привязка указателя d_err_1d к началу выделенной области памяти
    cudaMalloc(&d_err_1d, sizeof(double) * (size * size));
    
    //Объявление блока потоков для вычисления ошибки размером 1024x1x1
    dim3 errBS(1024, 1, 1);
    //Объявление сетки блоков потоков для вычисления ошибки размером,
    //вычисленным по формуле ceil((size * size) / (float)errBS.x) на каждую ось
    dim3 errGS(ceil((size * size) / (float)errBS.x), 1, 1);

    double* dev_out;
    //выделение памяти на устройстве для массива результатов вычисления ошибки
    //размером errGS.x элементов типа double и привязка указателя dev_out
    //к началу выделенной области памяти
    cudaMalloc(&dev_out, sizeof(double) * errGS.x);


    double* d_ptr;
    //Выделение памяти на устройстве для переменной типа double
    //и привязка указателя d_ptr к началу выделенной области памяти
    cudaMalloc((void**)(&d_ptr), sizeof(double));
    
    //Cинхронизация потоков на устройстве
    cudaDeviceSynchronize();
    while ((error_host > tol) && (iter_host < iter_max)) {
        iter_host++;

        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            error_host = 0.0;
            // Вычисляем разницу между A и Anew
            heatError << <GS, BS >> > (d_A, d_Anew, size, error_host, d_err_1d);
            // Уменьшаем размерность массива ошибки до одномерного массива
            errorReduce << <errGS, errBS, (errBS.x) * sizeof(double) >> > (d_err_1d, dev_out, size * size);
            errorReduce << <1, errBS, (errBS.x) * sizeof(double) >> > (dev_out, d_err_1d, errGS.x);
            cudaMemcpy(&error_host, &d_err_1d[0], sizeof(double), cudaMemcpyDeviceToHost);
        }
        
        //Вызываем функцию heat для вычисления новых значений Anew
        else heat << <GS, BS >> > (d_A, d_Anew, size);

        d_ptr = d_A;
        d_A = d_Anew;
        d_Anew = d_ptr;
    
        //Отслеживаем прогресс вычислений
        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            printf("%d : %lf\n", iter_host, error_host);
            fflush(stdout);
        }
    }
    
    //Вывод результата
    printf("%d : %lf\n", iter_host, error_host);
    return 0;
}
