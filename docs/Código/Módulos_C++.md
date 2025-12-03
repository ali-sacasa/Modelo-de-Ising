# Módulos en C++

La implementación en C++17 es el camino nativo y de alto rendimiento del proyecto, pero replica la misma lógica física y numérica del solver en Python. Se apoya en Eigen para álgebra lineal y se organiza en tres archivos:
- `hpp_Solver_D.E.s.hpp`: definiciones de tipos, clases y constantes.
- `Cpp_Solver_D.E.s.cpp`: implementación de todos los módulos numéricos.
- `Main_C++_Solver_D.E.s.cpp`: configuración, ejecución y guardado de resultados.


## Clases y responsabilidades

### `SystemConfig`
- Contiene parámetros físicos (N, J, h, periodic) y de muestreo o proyección (S, M_trunc).
- Define la malla temporal (`T`, `nt`, `t_list`) y los parámetros del flujo de Ricci (`BETA_MAX`, `n_beta`).
- Incluye los métodos `generate_time_list()` y `print_config()` para preparar y reportar la configuración.

### `QuantumOperators`
- Genera matrices de Pauli y la identidad de 2x2.
- Utilidades de productos de Kronecker (`kron_list` y `kron_list_sparse`).
- Construcción del Hamiltoniano de Ising transversal completo (`build_ising_hamiltonian`) y descompuesto en componentes ZZ/X (`build_ising_hamiltonian_components`).
- Operadores globales de espín (`build_Sz_total`, `build_Sx_total`) y lista de operadores de energía local (`build_local_energy_operators`).

### `CoherentStates`
- Construye estados coherentes de un solo espín (`coherent_state`).
- Muestrea puntos en la esfera de Bloch con un lattice de Fibonacci (`sample_cp1_fibonacci`).
- Genera los estados producto para N espines y normaliza las columnas (`build_product_coherent_vectors`).
- Calcula la matriz de Gram (`compute_gram_matrix`) y construye un proyector ortonormal truncado(`build_projection_from_gram`).

### `RicciFlow`
- Usa pesos térmicos $e^{-\beta H}$ para deformar la proyección hacia estados de baja energía (`get_thermal_weights`).
- Construye proyecciones ponderadas (`build_weighted_projection`) y evalúa la fidelidad con el estado inicial (`calculate_fidelity`).
- `optimize_projection` analiza una lista de betas, guarda la mejor proyección, el Hamiltoniano efectivo y las fidelidades, y exporta los resultados con `save_results_csv`.

### `ChebyshevPropagator`
- Reescala el Hamiltoniano y calcula coeficientes de Chebyshev para aproximar $e^{-iHt}$.
- `evolve` aplica la expansión a un estado dado. El `project_and_evolve` evoluciona en paralelo el estado proyectado y el completo y devuelve ambas series de estados.

### `EnergyAnalyzer`
- `compute_energy_evolution`:  computa la energía total a lo largo del tiempo.
- `compute_energy_components_evolution`: computa la descomposición ZZ y X.
- `compute_local_energy_density`: computa la densidad de energía por sitio.
- `compute_energy_variance`: computa la varianza de la energía.
- `compute_energy_conservation_error`: computa el error absoluto y relativo respecto a la energía inicial.

### `CSVExporter`
- Guarda vectores y matrices reales o complejas en CSV.
- Guardado específico de resultados de energía (`save_energy_results_csv`) y magnetización (`save_magnetization_csv`).

## Flujo por pasos (véase `Main_C++_Solver_D.E.s.cpp`)
1. Configura los parámetros físicos y numéricos (`SystemConfig`) y construye el Hamiltoniano (componentes ZZ y X).
2. Muestrea estados coherentes, arma la matriz de datos y ejecuta el flujo de Ricci para obtener el mejor proyector y Hamiltoniano efectivo.
3. Evoluciona el estado inicial en el subespacio proyectado y en el espacio completo mediante `ChebyshevPropagator`.
4. Calcula energías, magnetización, densidades locales, fidelidad y varianzas con `EnergyAnalyzer`.
5. Exporta todos los resultados en CSV con `CSVExporter` y reporta cálculos de errores.
La intención es que estos pasos y observables sean equivalentes a los del notebook en Python, facilitando comparar rendimiento y precisión entre ambos caminos.
