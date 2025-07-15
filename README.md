# 🎓 Asignación Óptima de Alumnos

Este proyecto resuelve el problema de asignación de alumnos desde cursos originales a nuevos cursos utilizando **Programación Lineal Entera** con la librería `PuLP`. Se busca minimizar un costo (probablemente relacionado a preferencias o distancias) cumpliendo restricciones como capacidad de los cursos y elección de los alumnos.

## 📌 Descripción del Problema

Dado:

* Un número total de alumnos (`tot_alum`)
* Varios cursos originales y nuevos
* Las preferencias de cada alumno por ciertos cursos nuevos

Se busca:

* Asignar a cada alumno a **exactamente un curso nuevo**
* Que se respete la capacidad de los cursos nuevos
* Que se respete las **preferencias de los alumnos**
* Que se minimice un costo asociado (por ejemplo, el nivel de preferencia)

## ⚙️ Tecnologías Utilizadas

* **Python 3**
* **[PuLP](https://coin-or.github.io/pulp/)** – para modelar y resolver el problema de optimización
* **Pandas / NumPy** – para la manipulación de datos y resultados

## 🧠 Estructura del Notebook

1. **Instalación de Dependencias**
   Instalación de `pulp` si no está previamente instalado.

2. **Generación de Datos Simulados**
   Se crean los datos de entrada: total de alumnos, cursos, preferencias, etc.

3. **Modelado del Problema**
   Se define el modelo de optimización:

   * Variables de decisión (asignación alumno-curso)
   * Función objetivo
   * Restricciones (asignación única, capacidad, etc.)

4. **Resolución del Modelo**
   Se utiliza el solucionador CBC para resolver el modelo.

5. **Visualización de Resultados**
   Se genera un DataFrame con las asignaciones finales.

## ▶️ Cómo Ejecutar

1. Clona este repositorio:

   ```bash
   git clone https://github.com/fcohenriquez/asign_alum
   cd asignacion-alumnos
   ```

2. Instala las dependencias necesarias:

   ```bash
   pip install pulp pandas numpy
   ```

3. Ejecuta el python `asign_alum.py`.
   
    ```bash
    ## Linux o MacOS
    python3 asign_alumn.py
  
    ## En Windows
    python asign_alumn.py
     ```


## 📈 Resultados Esperados

* Asignación de cada alumno a un curso nuevo de forma óptima.
* Visualización clara de qué alumno quedó en qué curso.

## 📚 Referencias

* [Documentación de PuLP](https://coin-or.github.io/pulp/)
* Modelos de asignación en Investigación Operativa

## Autor 🧠
[Francisco Henriquez](https://github.com/fcohenriquez) AND [Matias Henriquez](https://github.com/C5rsdMat1X5)
