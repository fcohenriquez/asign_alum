# -*- encoding: utf-8 -*-

'''
Author: fco henriquez
'''

# ============================================================
# Librerías principales
# ============================================================
import pandas as pd
import numpy as np
import random

pd.options.mode.chained_assignment = None
print("Setup OK")

"""
Módulo para la simulación y resolución del problema de asignación de alumnos a cursos.
Incluye generación de datos simulados, resolución mediante programación lineal y evaluación de resultados.

Autor: francisco.henriquez
Fecha de creación: 2022-01-20
"""

# ============================================================
# Sección: Parámetros personalizables de simulación
# ============================================================
# Modifique estos parámetros para personalizar la simulación según sus necesidades

TOTAL_ALUMNOS = 108  # Número total de alumnos a asignar
NUM_CURSOS_ORIG = 4  # Número de cursos originales de los alumnos
NUM_CURSOS_FINALES = 3  # Número de cursos finales a formar
NUM_ESCOGIDOS = 3  # Número de compañeros escogidos por alumno
PORC_INCOMP = 0.05  # Proporción de pares de alumnos incompatibles (0 a 1)
SEED_RANDOM = random.randint(1, 99)  # Semilla para la reproducibilidad de la simulación
MIN_NINAS_POR_CURSO = None  # Mínimo de niñas por curso final (None para automático)
MIN_ALUMNOS_POR_CURSO = None  # Mínimo de alumnos por curso final (None para automático)
MIN_ORIGEN_POR_CURSO = None  # Mínimo de alumnos de cada curso original por curso final (None para automático)
BALANCEAR_GENERO = True  # Si True, intenta balancear género en los cursos finales
BALANCEAR_ORIGEN = (
    True  # Si True, intenta balancear cursos de origen en los cursos finales
)
# ============================================================
# Sección: Generación de datos simulados para el problema
# ============================================================
def gen_alum_sim(tot_alum, num_cursos_orig=4, num_escogidos=3, por_incomp=0.05):
    """
    Genera un DataFrame simulado de alumnos con cursos originales, compañeros escogidos e incompatibilidades.    Parámetros:

        num_cursos_orig (int): Número de cursos originales.
        tot_alum (int): Número total de alumnos.
        por_incomp (float): Proporción de pares de alumnos incompatibles.
        num_escogidos (int): Número de compañeros escogidos por alumno.

    Retorna:
        pd.DataFrame: DataFrame con los datos simulados de alumnos.
    """
    # Determina la cantidad de pares incompatibles a generar
    num_par_imcomp = int(tot_alum * por_incomp)

    id_list = list(range(1, tot_alum + 1))

    # Asigna cursos originales de manera balanceada
    curso_orig = []
    for i in range(1, num_cursos_orig):
        curso_orig += [i] * int(round((tot_alum / num_cursos_orig)))
    curso_orig += [num_cursos_orig] * (tot_alum - len(curso_orig))

    # Asigna aleatoriamente el género femenino (1) o masculino (0)
    ninas = np.random.randint(0, 2, size=tot_alum).tolist()

    # Genera pares de alumnos incompatibles de forma aleatoria
    if num_par_imcomp > 0:
        incomp = random.sample(id_list, k=num_par_imcomp * 2)
        incompat_tp = list(zip(incomp[:num_par_imcomp], incomp[num_par_imcomp:]))
        incompatibles = pd.DataFrame(incompat_tp, columns=["inc_1", "inc_2"])
        inc2 = incompatibles.rename(columns={"inc_1": "inc_2", "inc_2": "inc_1"})
        incompatibles = pd.concat([incompatibles, inc2], ignore_index=True)
        incompatibles = incompatibles.rename(
            columns={"inc_1": "id", "inc_2": "incompatible"}
        )
        incompatibles["id"] = incompatibles["id"].astype("int64")
        incompatibles["incompatible"] = incompatibles["incompatible"].astype("int64")
    else:
        incompatibles = pd.DataFrame(columns=["id", "incompatible"])

    # Asigna compañeros escogidos para cada alumno
    escogidos_orig = [
        random.sample([x for x in id_list if x != _id], k=num_escogidos)
        for _id in id_list
    ]
    escogidos = np.array(escogidos_orig).T

    datos_entrada = pd.DataFrame(
        {"id": id_list, "curso_orig": curso_orig, "nina": ninas}
    )

    for i, escogido in enumerate(escogidos, start=1):
        datos_entrada[f"escogido_{i}"] = escogido

    # Une la información de incompatibilidades al DataFrame principal
    if not incompatibles.empty:
        datos_entrada = datos_entrada.merge(incompatibles, on="id", how="left")
    else:
        datos_entrada["incompatible"] = np.nan

    print("Generados los alumnos simulados")
    return datos_entrada


# ============================================================
# Sección: Resolución del problema de asignación de alumnos
# ============================================================


def asign_alum(
    datos_entrada,
    num_cursos=3,
    min_ninas_por_curso=None,
    min_alumnos_por_curso=None,
    min_origen_por_curso=None,
):
    """
    Asigna alumnos a cursos finales utilizando programación lineal entera.

    Parámetros:
        datos_entrada (pd.DataFrame): DataFrame con los datos de los alumnos.
        num_cursos (int): Número de cursos finales.

    Retorna:
        pd.DataFrame: DataFrame con la asignación óptima de alumnos a cursos.
    """

    import re
    from pulp import lpSum, LpProblem, LpVariable, LpMinimize, LpStatus, PULP_CBC_CMD

    # Extrae y prepara los datos necesarios desde el DataFrame de entrada
    tot_alum = len(datos_entrada)
    num_cursos_orig = len(datos_entrada["curso_orig"].drop_duplicates())
    tam_cursos = (
        min_alumnos_por_curso
        if min_alumnos_por_curso is not None
        else int((tot_alum // num_cursos))
    )
    max_mismo_curs_orig = (
        min_origen_por_curso
        if min_origen_por_curso is not None
        else int((tot_alum) // (num_cursos * num_cursos_orig))
    )
    ninas_por_curso = (
        min_ninas_por_curso
        if min_ninas_por_curso is not None
        else int(datos_entrada["nina"].sum() // num_cursos)
    )

    id = list(datos_entrada["id"])
    ninas = list(datos_entrada["nina"])
    curso_orig = list(datos_entrada["curso_orig"])
    n_incompat = int(len(datos_entrada[~datos_entrada["incompatible"].isna()]) / 2)
    df_inc = datos_entrada[~datos_entrada["incompatible"].isna()][
        ["id", "incompatible"]
    ]
    incompat_tp_0 = list(
        zip(list(df_inc["id"]), list(df_inc["incompatible"].astype("int64")))
    )
    incompat_tp_0 = [sorted(x) for x in incompat_tp_0]
    incompat_tp = set(tuple(x) for x in incompat_tp_0)
    incompat_tp = list(incompat_tp)

    num_escogidos = len(
        [
            x
            for x in [re.findall(r"escogido", x) for x in list(datos_entrada.columns)]
            if x != []
        ]
    )
    escogidos_orig = [
        list(x) for x in datos_entrada.iloc[:, -num_escogidos:-1].to_numpy()
    ]

    # Inicializa el modelo de optimización lineal entera
    model = LpProblem("Asign_alum", LpMinimize)

    # Define las variables de decisión: asignación de cada alumno a cada curso
    variable_names = [
        "cur_" + str(i) + "_alum_" + str(j)
        for j in range(1, tot_alum + 1)
        for i in range(1, num_cursos + 1)
    ]
    DV_variables = LpVariable.matrix("X", variable_names, cat="Integer", lowBound=0)
    rest_1 = np.array(DV_variables[0 : tot_alum * num_cursos]).reshape(
        tot_alum, num_cursos
    )

    # Define la función objetivo (minimizar suma de variables, dummy)
    obj_func = lpSum(DV_variables)
    model += obj_func

    # Restricción: cada alumno debe estar asignado a un solo curso
    for i in range(tot_alum):
        model += (
            lpSum(rest_1[i][j] for j in range(num_cursos)) == 1,
            "rest_un_solo_alumno " + str(i),
        )

    # Restricción: cada curso debe tener al menos el tamaño mínimo de alumnos
    for i in range(num_cursos):
        model += (
            lpSum(rest_1[j][i] for j in range(tot_alum)) >= tam_cursos,
            "rest de alumnos por curso " + str(i),
        )

    # Restricción: cada curso debe tener al menos el mínimo de niñas
    for i in range(num_cursos):
        model += (
            lpSum(rest_1[j][i] * ninas[j] for j in range(tot_alum)) >= ninas_por_curso,
            "ninas por curso " + str(i),
        )

    # Restricción: distribución equitativa de alumnos de cursos originales
    for c in range(1, num_cursos_orig + 1):
        for i in range(num_cursos):
            en_co = [1 if x == c else 0 for x in curso_orig]
            model += (
                lpSum(rest_1[j][i] * en_co[j] for j in range(tot_alum))
                >= max_mismo_curs_orig,
                "curso orig " + str(c) + " en curso " + str(i),
            )

    # Restricción: alumnos incompatibles no pueden estar en el mismo curso
    for c in range(n_incompat):
        for i in range(num_cursos):
            incomp = [
                1 if x == incompat_tp[c][0] or x == incompat_tp[c][1] else 0 for x in id
            ]
            model += (
                lpSum(rest_1[j][i] * incomp[j] for j in range(tot_alum)) <= 1,
                "incompatibles " + str(c) + " en el curso " + str(i),
            )

    # Restricción: cada alumno debe estar al menos con uno de sus escogidos
    for i in range(tot_alum):
        escog = [1 if x in escogidos_orig[i] else 0 for x in id]
        for k in range(num_cursos):
            model += (
                lpSum(rest_1[j][k] * escog[j] for j in range(tot_alum)) >= rest_1[i][k],
                "escogidos alumno" + str(i) + " en el curso " + str(k),
            )

    # Resuelve el modelo de optimización
    model.solve(PULP_CBC_CMD())

    status = LpStatus[model.status]
    print("Status de la optimizacion: " + status)

    if status == "Optimal":
        print("👌")
    else:
        print("☹️")

    # Procesa las variables de decisión para obtener la asignación final
    DF_result0 = pd.DataFrame()
    result_rows = []
    for v in model.variables():
        try:
            if v.value() is not None and v.value() > 0:
                result_rows.append({"nombre": v.name, "valor": v.value()})
        except Exception as e:
            print(f"error couldnt find value: {e}")

    DF_result0 = pd.DataFrame(result_rows)

    if DF_result0.empty:
        raise ValueError(
            "No se encontraron soluciones óptimas para las variables de decisión."
        )

    DF_result0["curso"] = DF_result0["nombre"].str.extract(r"cur_(\d+)_")[0].astype(int)
    DF_result0["id"] = DF_result0["nombre"].str.extract(r"alum_(\d+)$")[0].astype(int)
    DF_result0 = DF_result0.drop(columns=["nombre", "valor"])

    # Une la asignación con los datos originales
    DF_result0 = DF_result0.merge(datos_entrada, on="id", how="outer")
    DF_result0 = DF_result0.sort_values(by=["curso", "id"])
    return DF_result0


# ============================================================
# Sección: Evaluación de la asignación obtenida
# ============================================================


def eval_asig(
    df_result0,
    min_ninas_por_curso=None,
    min_alumnos_por_curso=None,
    min_origen_por_curso=None,
):
    """
    Evalúa el resultado de la asignación de alumnos a cursos.

    Parámetros:
        df_result0 (pd.DataFrame): DataFrame con la asignación de alumnos.

    Imprime:
        Resultados de la evaluación de duplicados, distribución de alumnos, niñas,
        cursos originales, incompatibles y escogidos.
    """
    import re
    import pandas as pd

    print("=" * 60)
    print("Evaluación de los resultados".center(60))
    print("=" * 60)

    DF_result = df_result0.copy()

    tot_alum = len(DF_result)
    num_cursos_orig = len(DF_result["curso_orig"].drop_duplicates())
    num_cursos = len(DF_result["curso"].drop_duplicates())
    tam_cursos = (
        min_alumnos_por_curso
        if min_alumnos_por_curso is not None
        else int((tot_alum // num_cursos))
    )
    max_mismo_curs_orig = (
        min_origen_por_curso
        if min_origen_por_curso is not None
        else int((tot_alum) // (num_cursos * num_cursos_orig))
    )
    ninas_por_curso = (
        min_ninas_por_curso
        if min_ninas_por_curso is not None
        else int(DF_result["nina"].sum() // num_cursos)
    )
    num_escogidos = len(
        [
            x
            for x in [re.findall(r"escogido", x) for x in list(DF_result.columns)]
            if x != []
        ]
    )

    # 1. Verifica duplicados por alumno
    print("\n[1] Duplicados por alumno:")
    print(DF_result["id"].duplicated().value_counts())
    if len(DF_result[DF_result["id"].duplicated(keep=False)]) == 0:
        print("✔️ No hay duplicados.")
    else:
        print("❌ Hay duplicados:")
        print(DF_result[DF_result["id"].duplicated(keep=False)].to_string(index=False))

    # 2. Muestra la cantidad de alumnos por curso
    print("\n[2] Cantidad de alumnos por curso:")
    alumnos_curso = DF_result.groupby("curso")["id"].count()
    print(alumnos_curso.to_string())
    if alumnos_curso.min() >= tam_cursos:
        print("✔️ Todos los cursos cumplen el mínimo de alumnos.")
    else:
        print("❌ Algún curso no cumple el mínimo.")

    # 3. Muestra la cantidad de niñas por curso
    print("\n[3] Cantidad de niñas por curso:")
    ninas_curso = DF_result.groupby("curso")["nina"].sum()
    print(ninas_curso.to_string())
    if ninas_curso.min() >= ninas_por_curso:
        print("✔️ Todos los cursos cumplen el mínimo de niñas.")
    else:
        print("❌ Algún curso no cumple el mínimo de niñas.")

    # 4. Muestra la distribución de cursos originales por curso final
    print("\n[4] Distribución de cursos originales por curso:")
    tabla_cursos = pd.crosstab(DF_result["curso"], DF_result["curso_orig"])
    print(tabla_cursos)
    if tabla_cursos.min().min() >= max_mismo_curs_orig:
        print("✔️ Todos los cursos cumplen el mínimo de alumnos por curso original.")
    else:
        print("❌ Algún curso no cumple el mínimo de alumnos por curso original.")

    # 5. Muestra la tabla de incompatibles asignados
    print("\n[5] Tabla de incompatibles asignados (solo los que tienen incompatibles):")
    incompatibles_tabla = DF_result[~DF_result["incompatible"].isna()][
        ["id", "curso", "incompatible"]
    ]
    if not incompatibles_tabla.empty:
        incompatibles_tabla = incompatibles_tabla.astype({"incompatible": "int64"})
        incompatibles_tabla = incompatibles_tabla.sort_values(by=["curso", "id"])
        print(incompatibles_tabla.to_string(index=False))
    else:
        print("No hay incompatibles asignados.")

    # 6. Evalúa si hay incompatibles en el mismo curso
    print("\n[6] Evaluación de incompatibles en el mismo curso:")
    eval_incomp = incompatibles_tabla.copy()
    if not eval_incomp.empty:
        eval_incomp_aux = eval_incomp[["id", "curso"]]
        eval_incomp_aux.columns = ["incompatible", "curso_incomp"]
        eval_incomp = eval_incomp.merge(eval_incomp_aux, on="incompatible", how="left")
        eval_incomp["prob_incomp"] = eval_incomp["curso"] == eval_incomp["curso_incomp"]
        print(
            eval_incomp[
                ["id", "incompatible", "curso", "curso_incomp", "prob_incomp"]
            ].to_string(index=False)
        )
        if len(eval_incomp[eval_incomp["prob_incomp"]]) == 0:
            print("✔️ No hay incompatibles en el mismo curso.")
        else:
            print("❌ Hay incompatibles en el mismo curso:")
            print(
                eval_incomp[eval_incomp["prob_incomp"]][
                    ["id", "incompatible", "curso"]
                ].to_string(index=False)
            )
    else:
        print("No hay incompatibles para evaluar.")

    # 7. Muestra la tabla de escogidos por alumno
    print("\n[7] Tabla de escogidos por alumno:")
    cols_escogidos = ["id", "curso"] + [
        f"escogido_{x}" for x in range(1, num_escogidos + 1)
    ]
    if all(col in DF_result.columns for col in cols_escogidos):
        print(
            DF_result[cols_escogidos]
            .sort_values(by=["curso", "id"])
            .to_string(index=False)
        )
    else:
        print("No hay columnas de escogidos para mostrar.")

    # 8. Evalúa si los alumnos están con al menos uno de sus escogidos
    print("\n[8] Evaluación de escogidos:")
    DF_escogidos = DF_result[
        ["id", "curso"] + ["escogido_" + str(x) for x in range(1, num_escogidos + 1)]
    ]
    curso_esc = DF_result[["id", "curso"]]
    DF_escogidos["eval_esc"] = False
    for e in range(1, num_escogidos + 1):
        curso_esc.columns = ["escogido_" + str(e), "curso_e" + str(e)]
        DF_escogidos = DF_escogidos.merge(
            curso_esc, on="escogido_" + str(e), how="left"
        )
        DF_escogidos.loc[
            DF_escogidos["curso"] == DF_escogidos["curso_e" + str(e)], "eval_esc"
        ] = True
    print(
        DF_escogidos[
            ["id", "curso"]
            + [f"escogido_{x}" for x in range(1, num_escogidos + 1)]
            + ["eval_esc"]
        ].to_string(index=False)
    )
    if len(DF_escogidos[DF_escogidos["eval_esc"]]) == len(DF_escogidos):
        print("✔️ Todos los alumnos están con al menos un escogido.")
    else:
        print("❌ Hay alumnos sin escogidos en su curso:")
        print(
            DF_escogidos[~DF_escogidos["eval_esc"]][
                ["id", "curso"] + [f"escogido_{x}" for x in range(1, num_escogidos + 1)]
            ].to_string(index=False)
        )

    # Resumen final de la evaluación
    print("\n" + "=" * 60)
    if (
        len(DF_result[DF_result["id"].duplicated(keep=False)]) == 0
        and alumnos_curso.min() >= tam_cursos
        and ninas_curso.min() >= ninas_por_curso
        and tabla_cursos.min().min() >= max_mismo_curs_orig
        and (eval_incomp.empty or len(eval_incomp[eval_incomp["prob_incomp"]]) == 0)
        and len(DF_escogidos[DF_escogidos["eval_esc"]]) == len(DF_escogidos)
    ):
        print("¡Se cumplieron todas las restricciones! 🎉".center(60))
    else:
        print("El resultado es subóptimo, revisar detalles arriba.".center(60))
    print("=" * 60)


def main():
    # Genera los datos simulados de entrada
    datos_entrada = gen_alum_sim(
        TOTAL_ALUMNOS,
        num_cursos_orig=NUM_CURSOS_ORIG,
        num_escogidos=NUM_ESCOGIDOS,
        por_incomp=PORC_INCOMP,
    )

    # Resuelve el problema de asignación de alumnos a cursos
    result = asign_alum(
        datos_entrada,
        num_cursos=NUM_CURSOS_FINALES,
        min_ninas_por_curso=MIN_NINAS_POR_CURSO,
        min_alumnos_por_curso=MIN_ALUMNOS_POR_CURSO,
        min_origen_por_curso=MIN_ORIGEN_POR_CURSO,
    )

    # Evalúa la asignación obtenida
    eval_asig(
        result,
        min_ninas_por_curso=MIN_NINAS_POR_CURSO,
        min_alumnos_por_curso=MIN_ALUMNOS_POR_CURSO,
        min_origen_por_curso=MIN_ORIGEN_POR_CURSO,
    )

    # Muestra la asignación final de alumnos por curso
    print("\nAsignación final de alumnos por curso:")
    for curso, grupo in result.groupby("curso"):
        print(f"\nCurso {curso}:")
        print(grupo[["id", "curso_orig", "nina"]].to_string(index=False))


if __name__ == "__main__":
    main()
