# Modelo de Ising

Implementación en C++ de un solver simplificado para el modelo de Ising cuántico con utilidades para generar y graficar salidas en CSV y PNG.

## Contenido del proyecto
- `solver_C++/Cpp_Solver_D.E.s.cpp`: Lógica principal del solver y utilidades numéricas.
- `solver_C++/Main_C++_Solver_D.E.s.cpp`: Punto de entrada que configura el sistema y ejecuta la simulación.
- `solver_C++/hpp_Solver_D.E.s.hpp`: Definiciones de tipos y clases.
- `solver_C++/run_solver.sh`: Script que compila, ejecuta y grafica la simulación.
- `solver_C++/plot_results.py`: Script de Python que genera la gráfica a partir de los CSV.
- `solver_C++/outputs/`: Directorio donde se guardan los archivos generados (`tiempos.csv`, `magnetizacion.csv`, `magnetizacion.png`).

## Requisitos
- Compilador C++17 (`g++` o equivalente).
- Python 3 con `matplotlib` instalado (`python3 -m pip install matplotlib`).

## Ejecución rápida
```bash
cd solver_C++
./run_solver.sh
```
El script compila y corre el solver, genera los CSV y produce la gráfica `outputs/magnetizacion.png`, mostrando la ventana con la figura.
