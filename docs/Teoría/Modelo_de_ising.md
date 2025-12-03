# Modelo de Ising

Este documento explica a grandes rasgos **modelo de Ising estándar (1D y 2D)**, útil tanto para física estadística como para conexiones con geometría y sistemas cuánticos.

---

## 1. Introducción al modelo de Ising

El modelo de Ising describe un sistema de **espines** que solo pueden tomar dos valores

$$
\sigma_i = \pm 1
$$

ubicados en los vértices de una malla de $N$ dimensiones. El Hamiltoniano clásico es

$$
H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i
$$

donde $J$ es el acoplamiento, $h$ es el campo externo y $\langle i,j \rangle$ denota pares vecinos.

---

## 2. Significado físico

Si $J>0$, los espines tienden a alinearse, como ocurre en el fenómeno de ferromagnetismo, si $J<0$ prefieren anti-alinearse, lo que refleja antiferromagnetismo. El campo $h$ busca orientar todos los espines en un mismo sentido, compitiendo con las interacciones internas.

---

## 3. Significado geométrico

### 3.1. Configuraciones como Vértices de un Hipercubo

Para $N$ espines, el espacio de configuraciones forma un hipercubo de dimensión $N$. El modelo estudia cómo la energía define un paisaje sobre ese hipercubo, generando regiones de baja y alta energía.

### 3.2. Curvatura Discreta

La energía

$$
E(\sigma) = -J\sum_{\langle i,j \rangle}\sigma_i\sigma_j
$$

favorece las configuraciones alineadas. Los bordes entre espines distintos actúan como defectos. En 2D estas fronteras son curvas cuya longitud aporta energía. Las configuraciones ordenadas presentan baja curvatura, mientras que configuraciones desordenadas muestran alta curvatura.

---

## 4. Significado termodinámico

A una temperatura $T$ cada configuración tiene peso

$$
P(\sigma) = \frac{1}{Z} e^{-\beta E(\sigma)}, \qquad \beta = \frac{1}{k_B T}.
$$

A bajas temperaturas predominan los mínimos energéticos, mientras que a altas temperaturas todas las configuraciones tienen pesos parecidos y predominan estados de desorden.

---

## 5. Soluciones Clásicas

### 5.1. Ising en 1D (sin campo)

Tiene solución exacta y no presenta transición de fase a temperatura finita. La magnetización

$$
M = \frac{1}{N}\sum_i \sigma_i
$$

tiende a cero en el límite termodinámico cuando $T>0$.

### 5.2. Ising en 2D (sin campo)

Existe una transición de fase a temperatura crítica $T_c$. Para $T < T_c$ y $M \neq 0$ (orden ferromagnético), para $T > T_c$ y $M = 0$ (desorden). La transición se refleja en el cambio de geometría de las interfaces entre dominios.

---

## 6. Dinámica Estocástica

La evolución clásica puede implementarse mediante dinámica de Glauber (cambios locales) o Kawasaki (conserva magnetización total). Son flujos estocásticos sobre el hipercubo que deforman interfaces y muestrean el paisaje energético.

---

## 7. Modelo cuántico

El Hamiltoniano cuántico estándar es

$$
H_Q = -J\sum_i \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x.
$$

El primer término define dominios y fronteras, el segundo introduce fluctuaciones cuánticas. Se estudian espectros, correlaciones y magnetización cuántica.
