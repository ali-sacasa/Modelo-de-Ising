# Modelo de Ising Usual: Una Introducción Pedagógica y Geométrica

Este documento explica el **modelo de Ising estándar (1D y 2D)** de manera clara, intuitiva y con una interpretación geométrica moderna, útil tanto para física estadística como para conexiones con geometría y sistemas cuánticos.

---

# 1. ¿Qué es el Modelo de Ising?

El modelo de Ising es uno de los modelos más fundamentales de la física teórica. Describe un sistema de **espines** que solo pueden tomar dos valores:
[
\sigma_i = \pm 1.
]
Estos espines están ubicados en los vértices de una red (1D, 2D o más dimensiones) y pueden interactuar entre sí.

La versión clásica no cuántica del Hamiltoniano es:
[
H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i.
]

Donde:

* (J) es la fuerza de acoplamiento entre espines.
* (h) es un campo magnético externo.
* (\langle i,j \rangle) denota pares vecinos.

---

# 2. Intuición Física

El modelo captura cómo los espines (pequeños imanes) **prefieren alinearse o anti-alinearse**.

* Si **(J > 0)**: los espines prefieren alinearse (ferromagnetismo).
* Si **(J < 0)**: prefieren anti-alinearse (antiferromagnetismo).

El campo externo (h) intenta orientar todos los espines en un mismo sentido.

---

# 3. Interpretación Geométrica

Aunque el modelo de Ising parece puramente combinatorio, tiene una interpretación geométrica profunda:

### 3.1. Configuraciones como Vértices de un Hipercubo

Cada espín (\sigma_i = \pm 1) puede verse como una coordenada en un espacio discreto. Para (N) espines:

* todo el espacio de configuraciones es un **hipercubo de dimensión (N)**.
* cada configuración es un punto ((\sigma_1,\ldots,\sigma_N)).

Geométricamente, el modelo de Ising estudia **cómo una energía asigna un paisaje geométrico** sobre este hipercubo.

---

### 3.2. Curvatura Discreta

La energía
[
E(\sigma) = -J\sum_{\langle i,j \rangle}\sigma_i\sigma_j
]
premia configuraciones alineadas.

Los bordes donde espines vecinos difieren son como **defectos cuasi-geométricos**, análogos a líneas de curvatura o bordes de dominio.

En 2D, estas fronteras entre regiones de espines (+) y (-) pueden interpretarse como **curvas sobre la red**, cuya longitud contribuye a la energía.

* Configuraciones ordenadas → baja “curvatura” (pocos bordes).
* Configuraciones desordenadas → alta “curvatura”.

Esto conecta el modelo de Ising con problemas geométricos discretos como:

* longitud de fronteras,
* minimización de área,
* interfaces mínimas.

---

# 4. Interpretación Termodinámica

A temperatura (T), cada configuración tiene peso:
[
P(\sigma) = \frac{1}{Z} e^{-\beta E(\sigma)}, \qquad \beta = \frac{1}{k_B T}.
]

### A bajas temperaturas:

* el sistema se congela en los mínimos de energía,
* domina el orden ferromagnético si (J>0).

### A altas temperaturas:

* todas las configuraciones pesan más o menos lo mismo,
* domina el desorden.

---

# 5. Soluciones Clásicas

## 5.1. Ising en 1D (sin campo)

Tiene solución exacta y **no** presenta transición de fase a temperatura finita.

El parámetro de orden (magnetización):
[
M = \frac{1}{N}\sum_i \sigma_i
]
es siempre cero en el límite termodinámico cuando (T>0).

---

## 5.2. Ising en 2D (sin campo)

Fue resuelto por Onsager. Aquí sí aparece una **transición de fase** a temperatura crítica (T_c).

De forma cualitativa:

* para (T < T_c): magnetización espontánea (M \neq 0),
* para (T > T_c): estado desordenado (M = 0).

Geométricamente, la transición corresponde a que las líneas de frontera entre regiones (+) y (-) cambian de comportamiento:

* por debajo de (T_c): regiones grandes y estables (interfaces suaves),
* por encima de (T_c): interfaces fractales y dominantes.

---

# 6. Conexión con Dinámica Estocástica

El modelo usual se dinamiza con:

* dinámica de Glauber
* dinámica de Kawasaki

Estas corresponden geométricamente a:

* flujos estocásticos sobre el estado del hipercubo,
* reglas locales que deforman las interfaces.

---

# 7. Conexión con el Modelo Cuántico (Contexto para tus notas)

El modelo de Ising clásico es el punto de partida para sus extensiones cuánticas:
[
H_Q = -J\sum_i \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x.
]

La parte clásica es exactamente la primera suma con (\sigma_i^z\sigma_{i+1}^z). Esa estructura geométrica de dominios y fronteras aparece nuevamente en:

* espectros,
* estados fundamentales,
* dinámica cuántica,
* subvariedades de baja energía.

---

# 8. Resumen Intuitivo

El modelo de Ising clásico describe:

* espines que quieren alinearse,
* geometría de interfaces y dominios,
* una energía que mide curvatura discreta,
* una transición de fase en 2D,
* un paisaje energético sobre un hipercubo.

Es la base geométrica, estadística y física del modelo cuántico que estás desarrollando.

---
