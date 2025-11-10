import sys
import random

def generate_graph(n_size, density_perc, inf_val, output_file):  
    print(f"Gerando grafo {n_size}x{n_size} com densidade {density_perc}%...")
    
    #Probabilidade de NÃO ter uma aresta
    inf_prob = (100 - density_perc) / 100.0
    
    with open(output_file, 'w') as f:
        for i in range(n_size):
            line_values = []
            for j in range(n_size):
                if i == j:
                    line_values.append(0)  #Custo 0 para a diagonal
                else:
                    #Decide se existe uma aresta
                    if random.random() < inf_prob:
                        line_values.append(inf_val) # Sem aresta
                    else:
                        #Com aresta, peso de 1 a 10
                        line_values.append(random.randint(1, 10))

            f.write(' '.join(map(str, line_values)) + '\n')
            
    print(f"Arquivo '{output_file}' gerado com sucesso.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python generate_graph.py <N> <Densidade> <Arquivo_Saida>")
        print("Exemplo: python generate_graph.py 1024 75 grafo_1024_75.txt")
        sys.exit(1)
        
    N = int(sys.argv[1])
    DENSIDADE = int(sys.argv[2])
    ARQUIVO = sys.argv[3]
    INF = 9999999
    
    generate_graph(N, DENSIDADE, INF, ARQUIVO)