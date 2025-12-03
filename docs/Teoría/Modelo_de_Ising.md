# Modelo de Ising Usual: Una Introducción Pedagógica y Geométrica

Este documento explica el **modelo de Ising estándar (1D y 2D)** de manera clara, intuitiva y con una interpretación geométrica moderna, útil tanto para física estadística como para conexiones con geometría y sistemas cuánticos.

---

## 1. ¿Qué es el Modelo de Ising?

El modelo de Ising es uno de los modelos más fundamentales de la física teórica. Describe un sistema de **espines** que solo pueden tomar dos valores

$$
\sigma_i = \pm 1
$$

Estos espines están ubicados en los vértices de una red 1D, 2D o más dimensiones y pueden interactuar entre sí. La versión clásica del Hamiltoniano es

$$
H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i
$$

donde

* $$J$$ es la fuerza de acoplamiento entre espines,
* $$h$$ es un campo magnético externo,
* $$\langle i,j \rangle$$ denota pares vecinos.

---

## 2. Intuición Física

El modelo captura cómo los espines (pequeños imanes) **prefieren alinearse o anti-alinearse**. Si $$J>0$$, los espines tienden a alinearse (ferromagnetismo), mientras que si $$J<0$$ prefieren anti-alinearse (antiferromagnetismo). El campo externo $$h$$ intenta orientar todos los espines en un mismo sentido, introduciendo competencia con las interacciones internas.

---

## 3. Interpretación Geométrica

Aunque a primera vista el modelo de Ising parece puramente combinatorio, posee una interpretación geométrica profunda.

### 3.1. Configuraciones como Vértices de un Hipercubo

Cada espín $$\sigma_i = \pm 1$$ puede considerarse como una coordenada en un espacio discreto. Para $$N$$ espines, todo el espacio de configuraciones forma un **hipercubo de dimensión $$N$$**, y cada configuración corresponde a un punto

$$
(\sigma_1, \ldots, \sigma_N)
$$

en ese hipercubo. Geométricamente, el modelo estudia **cómo la energía asigna un paisaje sobre este hipercubo**, definiendo regiones de baja y alta energía.

### 3.2. Curvatura Discreta

La energía

$$
E(\sigma) = -J\sum_{\langle i,j \rangle}\sigma_i\sigma_j
$$

premia configuraciones alineadas. Los bordes donde espines vecinos difieren actúan como **defectos cuasi-geométricos**, análogos a líneas de curvatura o bordes de dominio. En 2D, estas fronteras entre regiones de espines (+) y (-) pueden interpretarse como **curvas sobre la red**, cuya longitud contribuye directamente a la energía. Configuraciones ordenadas presentan baja “curvatura” (pocos bordes), mientras que configuraciones desordenadas exhiben alta “curvatura”. Esta conexión permite relacionar el modelo con problemas geométricos discretos como la minimización de área o interfaces mínimas.

---

## 4. Interpretación Termodinámica

A temperatura $$T$$, cada configuración tiene peso probabilístico

$$
P(\sigma) = \frac{1}{Z} e^{-\beta E(\sigma)}, \qquad \beta = \frac{1}{k_B T}
$$

donde $$Z$$ es la función de partición. A bajas temperaturas, el sistema se congela en los mínimos de energía, predominando el orden ferromagnético si $$J>0$$. A altas temperaturas, todas las configuraciones tienen aproximadamente la misma probabilidad, y el desorden domina.

---

## 5. Soluciones Clásicas

### 5.1. Ising en 1D (sin campo)

El modelo 1D tiene solución exacta y **no presenta transición de fase** a temperatura finita. La magnetización, definida como

$$
M = \frac{1}{N}\sum_i \sigma_i
$$

tiende a cero en el límite termodinámico cuando $$T>0$$, reflejando la ausencia de orden a temperatura positiva.

### 5.2. Ising en 2D (sin campo)

El modelo 2D fue resuelto por Onsager y **sí presenta transición de fase** a una temperatura crítica $$T_c$$. Para $$T < T_c$$, la magnetización espontánea $$M \neq 0$$, mientras que para $$T > T_c$$ el sistema se encuentra en un estado desordenado $$M = 0$$. Geométricamente, la transición corresponde a un cambio en las interfaces entre regiones de espines (+) y (-): por debajo de $$T_c$$, se forman regiones grandes y estables; por encima de $$T_c$$, las interfaces se fragmentan y dominan la configuración.

---

## 6. Conexión con Dinámica Estocástica

La evolución temporal del modelo clásico puede implementarse mediante:

* dinámica de **Glauber**, que permite cambios locales de espín,
* dinámica de **Kawasaki**, que conserva magnetización total.

Geométricamente, estas dinámicas corresponden a flujos estocásticos sobre el hipercubo de configuraciones, deformando progresivamente las interfaces y explorando el paisaje energético.

---

## 7. Extensión al Modelo Cuántico

El modelo de Ising clásico sirve como base para la extensión cuántica:

$$
H_Q = -J\sum_i \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x.
$$

Aquí, la primera suma corresponde a la parte clásica que define dominios y fronteras, mientras que la segunda introduce fluctuaciones cuánticas a través del operador de Pauli $$\sigma^x$$. Esta estructura permite estudiar:

* espectros y estados fundamentales,
* dinámica cuántica sobre subvariedades de baja energía,
* evolución de correlaciones y magnetización cuántica.

---

## 8. Resumen

El modelo de Ising clásico y cuántico describe

* espines que buscan alinearse,
* geometría de interfaces y dominios,
* un paisaje energético sobre un hipercubo,
* transiciones de fase en 2D,
* dinámicas estocásticas que exploran configuraciones,
* y, en su extensión cuántica, la evolución sobre subespacios de baja energía.

Esta interpretación combina física estadística, geometría discreta y cuántica, ofreciendo un marco intuitivo y computacionalmente relevante para entender sistemas de muchos cuerpos.
