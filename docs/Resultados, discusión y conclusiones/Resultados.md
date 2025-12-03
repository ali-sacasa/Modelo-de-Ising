# Resultados

Los resultados que entrega el binario C++ provienen de la configuración por defecto en `Main_C++_Solver_D.E.s.cpp`:
- $N = 8$ espines, $J = 1.0$, $h = 0.8$, condiciones abiertas (`periodic = false`).
- Muestreo de $S = 200$ estados coherentes, proyección truncada a `M_trunc = 30`.
- Malla temporal de `nt = 81` puntos hasta `T = 4.0` y orden de Chebyshev `cheb_order = 120`.
- Barrido de flujo de Ricci con `n_beta = 10` puntos hasta `BETA_MAX = 3.0`.

## Archivos generados
- `energy_results.csv`: energía total y sus componentes ZZ/X para el sistema completo y el proyectado.
- `magnetization.csv`: magnetización por sitio (normalizada por $N$) en el tiempo.
- `variance.csv`: varianza de la energía, útil para evaluar la estabilidad numérica.
- `local_energy_full.csv`, `local_energy_proj.csv`: densidad de energía por sitio para cada tiempo.
- `fidelity_evolution.csv`: fidelidad entre la trayectoria completa y la proyectada.
- `metrics.csv`: errores máximo y medio en energía y magnetización, fidelidades promedio e inicial.
- `ricci_beta_list.csv`, `ricci_fidelities.csv`: fidelidad obtenida para cada $\beta$ probada.
- `states_full_amplitudes.csv`, `states_proj_amplitudes.csv`: amplitudes (subconjunto de bases) para inspección.
- `config.csv`, `time.csv`: parámetros usados y malla temporal.

## Tendencias esperadas
- **Fidelidad:** la optimización con flujo de Ricci suele producir fidelidades iniciales altas (estado inicial bien capturado) y mantiene valores elevados durante la evolución para mallas temporales moderadas.
- **Energía y magnetización:** las curvas proyectadas siguen de cerca a las del espacio completo; las desviaciones se reflejan en `metrics.csv` y en la varianza.
- **Densidad de energía local:** permite identificar la formación de dominios o inhomogeneidades inducidas por el campo transversal.

Los notebooks en `docs/Código/Módulos_Python.md` incluyen ejemplos para graficar estas series y comparar la evolución completa vs. proyectada.
