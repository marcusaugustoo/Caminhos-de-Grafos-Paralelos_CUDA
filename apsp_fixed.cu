#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 
#include <errno.h>  
#include <cuda_runtime.h>


#define INF 9999999   
#define N 1024        //ALTERE AQUI PARA OS TESTES
#define BLOCK_SIZE 16 

//Função auxiliar para checar e reportar erros de CUDA.
static void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

/*
Kernel CUDA para uma iteração 'k' do Floyd-Warshall.
Lê da matriz de entrada 'd_in' e escreve o resultado em 'd_out'.
 */
__global__ void floyd_kernel(const int* d_in, int* d_out, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int ij_idx = i * n + j;
        int ik_idx = i * n + k;
        int kj_idx = k * n + j;

        int dij = d_in[ij_idx];
        int dik = d_in[ik_idx];
        int dkj = d_in[kj_idx];

        //Checa por INF antes de somar para evitar overflow
        if (dik == INF || dkj == INF) {
            d_out[ij_idx] = dij; 
        } else {
            //Calcula o novo caminho passando por 'k'
            int new_dist = dik + dkj;
            
            //Escreve o mínimo 
            d_out[ij_idx] = (dij > new_dist) ? new_dist : dij;
        }
    }
}

//Versão sequencial (CPU) do Floyd-Warshall.
void floyd_cpu(int* dist, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                
                int ij_idx = i * n + j;
                int ik_idx = i * n + k;
                int kj_idx = k * n + j;

                int dik = dist[ik_idx];
                int dkj = dist[kj_idx];

                if (dik == INF || dkj == INF) continue;

                int new_dist = dik + dkj;
            
                //Atualiza a distância se o novo caminho for menor
                if (dist[ij_idx] > new_dist) {
                    dist[ij_idx] = new_dist;
                }
            }
        }
    }
}

//Verifica se os resultados da CPU e GPU são idênticos.

bool verify_results(int* cpu_res, int* gpu_res, int n) {
    for (int i = 0; i < n * n; i++) {
        if (cpu_res[i] != gpu_res[i]) {
            //Reporta o primeiro erro encontrado
            printf("Erro na verificação! Posição %d: CPU=%d, GPU=%d\n",
                   i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    
    //Validação dos argumentos de linha de comando
    if (argc != 2) {
        fprintf(stderr, "Uso: ./apsp_exec <arquivo_do_grafo>\n");
        return -1;
    }
    char* graph_filename = argv[1];

    printf("Iniciando APSP com N = %d\n", N);
    srand((unsigned)time(NULL));

    //Alocação de Memória (CPU) 
    size_t matrix_size = (size_t)N * N * sizeof(int);

    int* h_dist_cpu = (int*)malloc(matrix_size);  
    int* h_dist_in  = (int*)malloc(matrix_size);  
    int* h_dist_out = (int*)malloc(matrix_size); 

    if (!h_dist_cpu || !h_dist_in || !h_dist_out) {
        fprintf(stderr, "Falha ao alocar memória no host\n");
        return -1;
    }

    //Leitura do Grafo
    printf("Lendo grafo do arquivo: %s\n", graph_filename);
    FILE* fp = fopen(graph_filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo %s: %s\n", graph_filename, strerror(errno));
        return -1;
    }

    //Lê os dados do arquivo para a matriz de entrada
    for (int i = 0; i < N * N; i++) {
        if (fscanf(fp, "%d", &h_dist_in[i]) != 1) {
            fprintf(stderr, "Erro ao ler dados do arquivo (posição %d).\n", i);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);

    //Copia os dados lidos para a matriz de teste da CPU
    memcpy(h_dist_cpu, h_dist_in, matrix_size);

    //3. Execução CPU
    printf("Executando Floyd-Warshall na CPU...\n");
    clock_t cpu_start = clock();
    floyd_cpu(h_dist_cpu, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Tempo CPU: %.6f ms\n", cpu_time_ms);

    //Execução GPU
    printf("Executando Floyd-Warshall na GPU...\n");
    
    int *d_in = NULL, *d_out = NULL;
    checkCuda(cudaMalloc((void**)&d_in, matrix_size));
    checkCuda(cudaMalloc((void**)&d_out, matrix_size));
    checkCuda(cudaMemcpy(d_in, h_dist_in, matrix_size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    
    //Inicia o cronômetro
    checkCuda(cudaEventRecord(start, 0));

    for (int k = 0; k < N; k++) {
        floyd_kernel<<<gridSize, blockSize>>>(d_in, d_out, k, N);
        
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize()); 

        int* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    //Para o cronômetro
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop)); 

    //Calcula o tempo
    float gpu_time_ms = 0;
    checkCuda(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    printf("Tempo GPU: %.6f ms\n", gpu_time_ms);

    //5. Verificação e Resultados 

    checkCuda(cudaMemcpy(h_dist_out, d_in, matrix_size, cudaMemcpyDeviceToHost));

    //Verifica se CPU e GPU geraram a mesma matriz
    bool success = verify_results(h_dist_cpu, h_dist_out, N);
    if (success) printf("Sucesso! Os resultados da CPU e GPU são idênticos.\n");
    else printf("Falha! Os resultados são diferentes.\n");

    //Imprime o Speedup
    double speedup = cpu_time_ms / (double)gpu_time_ms;
    printf("Speedup (CPU/GPU): %.2f x\n", speedup);
    
    printf("\nCSV_DATA,%.6f,%.6f,%.2f\n", cpu_time_ms, (double)gpu_time_ms, speedup);

    //6. Limpeza
    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    free(h_dist_cpu);
    free(h_dist_in);
    free(h_dist_out);

    return 0;
}