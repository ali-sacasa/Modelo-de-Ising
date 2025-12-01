# Uso del proyecto

## 1. Ejecutar la simulación y graficar
Desde el directorio principal del repositorio:
```bash
cd solver_C++
./run_solver.sh
```
El script:
1. Compila el binario `quantum_solver` en `outputs/`.
2. Ejecuta la simulación y guarda `outputs/tiempos.csv` y `outputs/magnetizacion.csv`.
3. Llama a `plot_results.py` para generar y mostrar la gráfica, guardándola como `outputs/magnetizacion.png`.

Para cambiar el directorio de salida:
```bash
./run_solver.sh otra_carpeta
```

Para guardar la figura sin abrir una ventana gráfica:
```bash
python3 plot_results.py --data-dir outputs --no-show
```

## 2. Servir la documentación con MkDocs
Instala MkDocs (si no lo tienes):
```bash
python3 -m pip install mkdocs
```
Luego levanta el sitio:
```bash
mkdocs serve
```
Abre el navegador en la URL indicada (por defecto `http://127.0.0.1:8000`). Para generar la versión estática:
```bash
mkdocs build
```
