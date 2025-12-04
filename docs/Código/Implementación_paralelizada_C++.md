# Paralelización del código de C++
El código fue paralelizado utilizando OpenMP (API de memoria compartida), con un enfoque en las secciones más pesadas del mismo. La selección de las secciones a paralelizar se basó principalmente en tres criterios:
1) **Frecuencia de ejecución:** Priorizar funciones que se llaman repetidas veces.
2) **Costo computacional:** Enfocarse en operaciones que toman un tiempo prolongado en completarse.
3) **Independencia de datos:** Identificar iteraciones sin dependencias entre sí.

En general, se paralelizaron las secciones donde se justificaba sustancialmente el overhead de OpenMP. Al ejecutar el código con un solo hilo (N=8, S=400), se obtuvieron tiempos de ejecución distribuidos de la siguiente forma:

```
Tiempo Total: 4.67 segundos

Desglose:
  - Estados coherentes:  0.002 s  (0.04%)
  - Optimización flujo de Ricci: 4.264 s  (91.3%)  ← Bottleneck principal
  - Cálculo energías: 0.229 s  (4.9%)
  - Evolución temporal (Chebyshev):  0.176 s  (3.8%)
```

## Secciones paralelizadas
- **Construcción de estados coherentes (productos tensoriales)**
```cpp
#pragma omp parallel for schedule(dynamic)
for (int j = 0; j < S; ++j) {
    VectorXcd single = coherent_state(z_list[j]);
    VectorXcd psi = single;
    
    for (int i = 1; i < N; ++i) {
        int rows_psi = psi.rows();
        int rows_single = single.rows();
        VectorXcd kron_vec(rows_psi * rows_single);
        
        for (int r1 = 0; r1 < rows_psi; ++r1) {
            kron_vec.segment(r1 * rows_single, rows_single) = psi(r1) * single;
        }
        psi = kron_vec;
    }
    Data.col(j) = psi;
}
```
Aquí, cada estado coherente se construye mediante productos tensoriales sucesivos. La paralelización distribuye S estados coherentes entre los threads disponibles, pero no lo hace de manera equitativa. la cláusula `scheduling(dynamic)` hace que los hilos trabajen conforme se van desocupando, lo que implica un balanceo de carga más eficiente.
Para esta sección es importante notar que el tiempo de ejecución muy bajo. Limitaciones computacionales no nos permitieron aumentar mucho el valor de N, por lo que los cálculos para esta parte eran relativamente sencillos. El efecto de la paralelización se observa mejor para valores mayores de N.
Además de esto, otra nota importante es la razón por la que no se paralelizó sobre el índice i (el que recorre N). Esto es así porque al trabajar con productos tensoriales sucesivos, cada uno depende del resultado del anterior, por lo que dividir el trabajo entre varios hilos sería ineficiente ya que tendrían que esperar a que los anteriores terminen.

- **Optimización con flujo de Ricci**
```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < n_beta; ++i) {
    double beta = beta_list[i];
    
    std::vector<double> weights = get_thermal_weights(H, Data, beta);
    std::vector<double> eigenvalues;
    MatrixXcd P = build_weighted_projection(Data, config.M_trunc, weights, eigenvalues);
    double fid = calculate_fidelity(P, psi0);
    
    results.weights_list[i] = weights;
    results.fidelities[i] = fid;
    P_list[i] = P;
    eigenvalues_list[i] = eigenvalues;
}
```
Como se vio antes, esta sección representa el mayor "cuello de botella" del código, consumiendo aproximadamente 91% del tiempo total de ejecución. Esto ocurre porque cada iteración implica la diagonalización del Hamiltoniano completo y de la matriz de Gram ponderada, las cuales son operaciones muy costosas. Además de eso, también es importante notar la 
implementación del scheduling dinámico, que es crucial ya que distintos valores de beta pueden afectar el costo computacional de dichas operaciones, por lo que es más eficiente que los hilos trabajen en nuevas iteraciones una vez que terminen la anterior.

**Evolución temporal con Chebyshev:**
```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < nt; ++i) {
    double t = t_list[i];
    
    VectorXcd psi_proj_t = prop_eff.evolve(psi0_proj, t);
    VectorXcd psi_proj_lifted = P * psi_proj_t;
    states_proj.col(i) = psi_proj_lifted;
    
    VectorXcd psi_full_t = prop_full.evolve(psi0, t);
    states_full.col(i) = psi_full_t;
}
```
La evolución temporal utiliza la expansión de Chebyshev para calcular la exponencial matricial exp(-iHt) aplicada al estado inicial. Una característica fundamental de este enfoque es que cada tiempo evoluciona de manera completamente independiente desde el estado inicial psi0, en contraste con métodos de paso temporal como Runge-Kutta que requieren conocer 
el estado en el paso anterior. Esta independencia temporal hace que la paralelización sea completamente natural, es decir, que las iteraciones se pueden dividir sin la preocupación de esperar a que terminen iteraciones anteriores. El código calcula simultáneamente la evolución en el espacio completo y en el subespacio proyectado, permitiendo comparar la 
calidad de la aproximación. Con valores de nt entre 81 y 161 puntos temporales, y usando expansiones de Chebyshev de orden M=120, cada evaluación involucra múltiples multiplicaciones matriz-vector que, aunque individualmente no son extremadamente costosas, se acumulan rápidamente hasta alcanzar un costo computacional considerable.

**Cálculo de observables (energía, magnetización, etc):**
```cpp
// Energía total
#pragma omp parallel for
for (int i = 0; i < nt; ++i) {
    VectorXcd state = states.col(i);
    Complex energy = state.adjoint() * (H_dense * state);
    energies[i] = energy.real();
}

// Densidad local de energía
#pragma omp parallel for collapse(2)
for (int i = 0; i < nt; ++i) {
    for (int j = 0; j < N; ++j) {
        VectorXcd state = states.col(i);
        MatrixXcd h_local_dense = MatrixXcd(h_local_list[j]);
        Complex e_local = state.adjoint() * (h_local_dense * state);
        local_energy(i, j) = e_local.real();
    }
}
```
Una vez completada la evolución temporal, el código calcula diversos observables físicos para analizar la dinámica del sistema. Estas operaciones incluyen la energía total, sus componentes (interacción ZZ y campo transversal X), la magnetización, y la densidad local de energía. Cada observable requiere el cálculo de valores esperados de la forma 
$\langle \psi(t) | H | \psi(t) \rangle$ para cada paso temporal, operaciones que son completamente independientes entre sí (cada iteración en el tiempo depende solo del estado en el sistema en ese tiempo). La mayoría de estos cálculos utilizan scheduling estático por defecto, ya que el costo de cada producto matriz-vector es uniforme. Sin embargo, 
para la densidad local de energía se emplea la cláusula collapse(2), que fusiona los ciclos anidades en un único espacio de iteraciones. Esto es particularmente útil porque transforma nt × N iteraciones en un conjunto más grande de tareas independientes, lo que tiende a mejorar el balanceo cuando alguno de los índices es pequeño. Concretamente en 
nuestro caso, al usar nt = 81 y N = 8, paralelizar solo sobre el ciclo externo, usando, por ejemplo, 12 hilos, llevaría a un desbalance de carga, mientras que al usar collapse se obtiene un balance perfecto. Al igual que en el punto anterior, los cálculos de esta sección no son particularmente costosos, pero al ejecutarse múltiples veces alcanzan
un costo considerable.

