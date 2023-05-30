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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cub/cub.cuh>

#define BILLION 1000000000.0

#define MAX(x, y) (((x)>(y))?(x):(y))

#define ABS(x) ((x)<0 ? -(x): (x))

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

char ERROR_WITH_ARGS[] = ">>> Not enough args\n";
char ERROR_WITH_ARG_1[] = ">>> Incorrect first param\n";
char ERROR_WITH_ARG_2[] = ">>> Incorrect second param\n";
char ERROR_WITH_ARG_3[] = ">>> Incorrect third param\n";

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;

////////////////////////////////////////////////////////////////
// Функция fillBorders заполняет границы двумерного массива значениями, 
// полученными путем линейной интерполяции на основе переданных параметров (top, bottom, left, right). 
// Функция использует технологию параллельных вычислений на GPU с помощью CUDA и выполняется в блоках и нитях. 
// Количество нитей в блоке должно быть кратным числу 32 и не превышать 1024.
////////////////////////////////////////////////////////////////
__global__ void fillBorders(double *arr, double top, double bottom, double left, double right, int m) {

        // Выполняем линейную интерполяцию на границах массива
        int j = blockDim.x * blockIdx.x + threadIdx.x;

        if ((j > 0) && (j < m)) {
	    arr[IDX2F(1,j,m)] = arr[IDX2F(1,j+m,m)] = (arr[IDX2F(1,1,m)] + top*(j-1));   //top
            arr[IDX2F(m,j,m)] = arr[IDX2F(m,j+m,m)]  = (arr[IDX2F(m,1,m)] + bottom*(j-1)); //bottom
            arr[IDX2F(j,1,m)]  = arr[IDX2F(j,m+1,m)] = (arr[IDX2F(1,1,m)] + left*(j-1)); //left
            arr[IDX2F(j,m,m)] = arr[IDX2F(j,2*m,m)] = (arr[IDX2F(1,m,m)] + right*(j-1)); //right
        }
}

////////////////////////////////////////////////////////////////
// Вычислям среднее значение элемента массива arr, используя значения элементов, расположенных вокруг него. 
// Функция принимает на вход указатель на массив arr, индексы p и q,
// определяющие расстояние между элементами вокруг центрального элемента,
// и m - размерность массива
////////////////////////////////////////////////////////////////
__global__ void getAverage(double *arr, int p, int q, int m) {

        // Присваиваем ячейке среднее значение от креста, окружающего её

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        //Если условие выполнено, то функция вычисляет среднее значение элемента arr[i,j+p], 
        //используя значения элементов, расположенных вокруг него
        if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
        arr[IDX2F(i,j+p,m)] = 0.25 * (arr[IDX2F(i+1,j+q,m)]
                                        + arr[IDX2F(i-1,j+q,m)]
                                        + arr[IDX2F(i,j-1+q,m)]
                                        + arr[IDX2F(i,j+1+q,m)]);
        }
}

////////////////////////////////////////////////////////////////
// Вычитаем элементы двух массивов arr_a и arr_b и сохраняет результат в массиве arr_b.
// Используем макрос IDX2F для вычисления индекса элемента матрицы по его строке и столбцу
////////////////////////////////////////////////////////////////
__global__ void subtractArrays(const double *arr_a, double *arr_b, int m) {

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        //Работаем только с элементами, которые расположены внутри матрицы,
        //то есть не находятся на её границах
        if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
	        arr_b[IDX2F(i,j,m)] = ABS(arr_a[IDX2F(i,j,m)] - arr_a[IDX2F(i,j+m,m)]);
        }
}


int main(int argc, char *argv[]){
    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);    
    double delta, min_error;
    int m, iter_max;
    
    // Проверка ввода данных
    if (argc < 4){
        printf("%s\n", ERROR_WITH_ARGS);
        exit(1);
    } else{
        m = atoi(argv[1]); // Размер сетки
        if (m == 0){
            printf("%s\n", ERROR_WITH_ARG_1);
            exit(1);
        }
        iter_max = atoi(argv[2]); // Количество итераций
        if (iter_max == 0){
            printf("%s\n", ERROR_WITH_ARG_2);
            exit(1);
        }
        min_error = atof(argv[3]); // Точность
        if (min_error == 0){
            printf("%s\n", ERROR_WITH_ARG_3);
            exit(1);
        }
    }    

    int iter = 0;
    double err = min_error + 1;
    size_t size = 2 * m * m * sizeof(double);
    double *arr = (double*)malloc(size);

    for(int j = 1; j <= m; j++){
        for(int i = 1; i <= m; i++){
            arr[IDX2F(i,j,m)] = 0;
        }
    }
    
    arr[IDX2F(1,1,m)] = arr[IDX2F(1,m+1,m)] = CORNER_1;
    arr[IDX2F(1,m,m)] = arr[IDX2F(1,2*m,m)] = CORNER_2;
    arr[IDX2F(m,1,m)] = arr[IDX2F(m,m+1,m)] = CORNER_4;
    arr[IDX2F(m,m,m)] = arr[IDX2F(m,2*m,m)] = CORNER_3;
    
    // Коэффициенты для линейной интерполяции
    double top, bottom, left, right;

    top = (arr[IDX2F(1,m,m)] - arr[IDX2F(1,1,m)])/(m-1);
    bottom = (arr[IDX2F(m,m,m)] - arr[IDX2F(m,1,m)])/(m-1);
    left = (arr[IDX2F(m,1,m)] - arr[IDX2F(1,1,m)])/(m-1);
    right = (arr[IDX2F(m,m,m)] - arr[IDX2F(1,m,m)])/(m-1);

    cudaError_t cudaErr = cudaSuccess;
    double *d_A = NULL;
    cudaErr = cudaMalloc((void **)&d_A, size);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    double *d_B = NULL;
    cudaErr = cudaMalloc((void **)&d_B, size/2);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaErr = cudaMemcpyAsync(d_A, arr, size, cudaMemcpyHostToDevice, stream);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }
    
    // Это ядро ​​заполняет границы с помощью линейной интерполяции
    fillBorders<<<(m + 1024 - 1)/1024, 1024, 0, stream>>>(d_A, top, bottom, left, right, m);
    cudaErr = cudaMemcpyAsync(arr, d_A, size, cudaMemcpyDeviceToHost, stream);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }

    printf("\n");
    if (m == 13) {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                printf("%06.3lf ", arr[IDX2F(i,j,m)]);
            }
            printf("\n");
        }
    }

    int p = m, q = 0, flag = 1;
    double *h_buff = (double*)malloc(sizeof(double));
    double *d_buff = NULL;

    cudaErr = cudaMalloc((void**)&d_buff, sizeof(double));
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(cudaErr));
        exit(1);
    }
    
    dim3 grid((m + 32 - 1)/32 , (m + 32 - 1)/32);    
    dim3 block(32, 32);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Вызываем DeviceReduce здесь, чтобы проверить, сколько памяти нам нужно для временного хранилища
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m*m, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    nvtxRangePushA("Main loop");
    {
    while(iter < iter_max && flag) {
    	if(!graphCreated) {
    		// Здесь мы начинаем фиксировать вызовы ядра в graph перед их вызовом.
            // Это позволяет нам сократить накладные расходы на вызовы
    		cudaErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    		if (cudaErr != cudaSuccess) {
           		    fprintf(stderr, "Failed to start stream capture (error code %s)!\n", cudaGetErrorString(cudaErr));
            	    exit(1);
    		}
    		for (int i = 0; i < 100; i++) {
                //q и p выбирают, какой массив мы считаем новым, а какой — старым.
    			q = (i % 2) * m;
    			p = m - q;
    			getAverage<<<grid, block, 0, stream>>>(d_A, p, q, m);
    		}

            //Отслеживаем статусы
    		cudaErr = cudaStreamEndCapture(stream, &graph);
    		if (cudaErr != cudaSuccess) {
                        fprintf(stderr, "Failed to end stream capture (error code %s)!\n", cudaGetErrorString(cudaErr));
                        exit(1);
                    }
    		cudaErr = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    		if (cudaErr != cudaSuccess) {
                        fprintf(stderr, "Failed to instantiate cuda graph (error code %s)!\n", cudaGetErrorString(cudaErr));
                        exit(1);
                    }
    		graphCreated = true;    
    	}

    	cudaErr = cudaGraphLaunch(instance, stream);
    	if (cudaErr != cudaSuccess) {
                fprintf(stderr, "Failed to launch cuda graph (error code %s)!\n", cudaGetErrorString(cudaErr));
                exit(1);
            }
    	if (cudaErr != cudaSuccess) {
                fprintf(stderr, "Failed to synchronize the stream (error code %s)!\n", cudaGetErrorString(cudaErr));
                exit(1);
            }

    	// Проверяем ошибку каждые 100 итераций
    	iter += 100;

    	// Здесь мы вычисляем абсолютные значения различий массивов,
        // а затем находим максимальную разницу, т.е. ошибку
    	subtractArrays<<<grid, block, 0, stream>>>(d_A, d_B, m);
    	cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m*m, stream);
    	cudaErr = cudaMemcpyAsync(h_buff, d_buff, sizeof(double), cudaMemcpyDeviceToHost, stream);

    	if (cudaErr != cudaSuccess) {
                fprintf(stderr, "Failed to copy error back to host memory(error code %s)!\n", cudaGetErrorString(cudaErr));
                exit(1);
            }
    	err = *h_buff;
    	flag = err > min_error;
    }
    }

    nvtxRangePop();

    clock_gettime(CLOCK_REALTIME, &stop);
    delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;

    printf("Elapsed time %lf\n", delta);
    printf("Final result: %d, %0.8lf\n", iter, err);

    cudaErr = cudaMemcpy(arr, d_A, size, cudaMemcpyDeviceToHost);

    if (m == 13) {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                printf("%06.3lf ", arr[IDX2F(i,j,m)]);
            }
            printf("\n");
        }
    }
        
    free(arr);
    free(h_buff);

    cudaFree(d_buff);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp_storage);

    return 0;
}
