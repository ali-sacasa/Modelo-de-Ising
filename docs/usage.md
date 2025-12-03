# Uso del proyecto

## 1. Ejecutar la versión en Python 
1) Abrir `Solver_D.E.s.ipynb` en Jupyter.
2) Ejecutar todas las celdas: se construye el Hamiltoniano, se optimiza la proyección con flujo de Ricci, se propaga con polinomios de Chebyshev y se grafican energía, magnetización y fidelidad.
3) Ajustar parámetros como `N`, `J`, `h`, `S`, `M_trunc`, `T`, `nt` en las celdas de configuración para nuevos experimentos.


## 2. Para compilar y ejecutar la versión en C++
Ejemplo de compilación desde el root del repositorio:
```bash
g++ -std=c++17 -O2 \
  -I/usr/include/eigen3 \
  Main_C++_Solver_D.E.s.cpp Cpp_Solver_D.E.s.cpp -o quantum_solver
./quantum_solver
```
El ejecutable crea los siguientes archivos CSV en el directorio actual:
- `config.csv`, `time.csv`
- `energy_results.csv`
- `magnetization.csv`
- `variance.csv`
- `local_energy_full.csv`, `local_energy_proj.csv`
- `states_full_amplitudes.csv`, `states_proj_amplitudes.csv`
- `fidelity_evolution.csv`, `metrics.csv`
- `ricci_beta_list.csv`, `ricci_fidelities.csv`

## 3. Ajustar parámetros físicos y numéricos
Para ajustar parámetros, se puede editar `Main_C++_Solver_D.E.s.cpp` antes de compilar. En la sección de configuración (`SystemConfig config;`) se puede modificar:
- `N`, `J`, `h`, `periodic` para definir el modelo de Ising transversal.
- `S` para el número de estados coherentes y `M_trunc` para dimensión máxima del subespacio proyectado.
- `T`, `nt` y `cheb_order` para la evolución temporal con polinomios de Chebyshev.
- `BETA_MAX` y `n_beta` para la exploración del flujo de Ricci, los pesos térmicos.

## 4. Analizar resultados con notebooks
- `análisis_datos_generados_en_c++.ipynb` carga los CSV generados y produce gráficas de energía y magnetización.
- `modelo_analítico.ipynb` contiene referencias analíticas para comparar resultados.

