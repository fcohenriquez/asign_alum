# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:14:16 2022

@author: francisco.henriquez
"""



###################################################################################################################
# Problema asignacion alumnos
##################################################################################################

import pandas as pd
import numpy as np
import random

pd.options.mode.chained_assignment = None 

print('setup ok')


# Funcion que genera los cursos simulados
#########################################

def gen_alum_sim (tot_alum, num_cursos_orig=4, num_escogidos=3, por_incomp=0.05):
  '''
  Entrega un DF simulado con los parametros entregados.
  por defecto tiene 4 cursos originales, 3 finales y 3 escogidos
  Tiene un 5% de pares de incompatibles 
  '''
 
  num_par_imcomp=int(tot_alum*por_incomp) # Cantidad de pares de incompatibles
  
  id=list(range(1,tot_alum+1))
  
  curso_orig=[]
  for _ in range(1,num_cursos_orig):
    curso_orig=curso_orig+[_]*int(round((tot_alum/num_cursos_orig)))
  curso_orig=curso_orig+[num_cursos_orig]*(tot_alum-len(curso_orig))
  
  ninas=[]
  for _ in range(tot_alum):
    ninas=ninas+[random.randint(0,1)]
  
  incomp=random.sample(id, k=num_par_imcomp*2)
  incompat_tp=list(zip(incomp[0:num_par_imcomp], incomp[num_par_imcomp:]))

  incompatibles=np.array(incompat_tp)
  incompatibles=pd.DataFrame(incompatibles, columns=['inc_1', 'inc_2'])
  inc2=incompatibles[['inc_1', 'inc_2']]
  inc2.columns=['inc_2', 'inc_1']
  incompatibles=incompatibles.append(inc2, ignore_index=True)
  incompatibles.columns=['id', 'incompatible']
  incompatibles['id']=incompatibles['id'].astype('int64')
  incompatibles['incompatible']=incompatibles['incompatible'].astype('int64')

  # Companeros escogidos 
  escogidos_orig=[]
  for _ in id:
  
    escogidos_orig.append(random.sample([x for x in id if x != _], k=num_escogidos))
  
  escogidos=np.array(escogidos_orig)
  escogidos=escogidos.T
  
  datos_entrada=pd.DataFrame()
  datos_entrada['id'] =id
  datos_entrada['curso_orig'] =curso_orig
  datos_entrada['nina'] =ninas
  
  i=1
  for _ in escogidos:
    datos_entrada['escogido_'+str(i)]=_
    i=i+1
  
  
  datos_entrada=datos_entrada.merge(incompatibles, on='id', how='left')
  
  print('Generados los alumnos simulados')
  return(datos_entrada)


# Funcion del PPL
###################


def asign_alum(datos_entrada, num_cursos=3):
  '''
  Esta funcion entrega un DF con la asignacion a partir del DF con los datos de los cursos
  Requiere que el DF tenga los campos con los nombres adecuados (aunque se puede parametrizar)
  '''
  
  import re
  from pulp import lpSum, LpProblem, LpVariable, LpMinimize, LpStatus, PULP_CBC_CMD
  # obtencion de los datos a partir del DF
  tot_alum=len(datos_entrada)
  num_cursos_orig=len(datos_entrada['curso_orig'].drop_duplicates())
  
  tam_cursos=int((tot_alum//num_cursos))
  max_mismo_curs_orig=int((tot_alum)//(num_cursos*num_cursos_orig))
  ninas_por_curso=int(datos_entrada['nina'].sum()//num_cursos)
  
  id=list(datos_entrada['id'])
  ninas=list(datos_entrada['nina'])
  curso_orig=list(datos_entrada['curso_orig'])
  n_incompat=int(len(datos_entrada[~datos_entrada['incompatible'].isna()])/2)
  df_inc=datos_entrada[~datos_entrada['incompatible'].isna()][['id', 'incompatible']]
  incompat_tp_0=list(zip(list(df_inc['id']), list(df_inc['incompatible'].astype('int64'))))
  incompat_tp_0=[sorted(x) for x in incompat_tp_0]
  incompat_tp=set(tuple(x) for x in incompat_tp_0)
  incompat_tp=list(incompat_tp)
  
  num_escogidos=len([x for x in [re.findall(r"escogido", x ) for x in list(datos_entrada.columns)] if x!=[]])
  escogidos_orig=[list (x) for x in datos_entrada.iloc[:,-num_escogidos:-1].to_numpy()]
  
    # Iniciacion del modelo
  
  model=LpProblem('Asign_alum', LpMinimize)
  
  # Definicion de las variables de decision
  
  variable_names = ['cur_'+str(i)+'_alum_'+str(j) for j in range(1, tot_alum+1) for i in range(1,num_cursos+1)]
  
  DV_variables = LpVariable.matrix("X", variable_names, cat = "Integer", lowBound= 0 )
  
  rest_1=np.array(DV_variables[0:tot_alum*num_cursos]).reshape(tot_alum,num_cursos)

  
  # Funcion Objetivo
  
  '''
  Se genera el vector funcion objetivo con el numero de elementos como la combinacion
  de alumnos y cursos 
  '''
  
  obj_func = lpSum(DV_variables)
  model +=  obj_func
  
  
  # Restricciones
  
  #print('\nrest_tam_alum: cada alumno tiene que estar en un solo curso')
  #Esta restriccion indica que hay y un solo alumno por curso y que todos los alumnos esten en un curso
  
  for i in range(tot_alum):
      model+=(lpSum(rest_1[i][j] for j in range(num_cursos))==1, 'rest_un_solo_alumno '+str(i))
  
  
  #print('\nrest_tam_curso: cada curso tiene que tener un numero de alumnos parecido')
  #Esta restriccion es para hacer que todos los cursos tengan cantidades similares de alumnos (no son iguales porque el total de alumnos puede que no sea divisible por el tamano de los cursos)
  
  for i in range(num_cursos):
    model+=(lpSum(rest_1[j][i] for j in range(tot_alum))>=tam_cursos, 'rest de alumnos por curso '+str(i))
  
  #print('\nRestricciones de ninas')
  # (este tipo de restricciones se puede utilizar para distribuir proporcionadamente otros grupos como los de mejor desempeno academico, etc.)
  #Esta restriccion es para hacer que todos los cursos tengan cantidades similares de alumnos y alumnas (no son iguales porque el total de alumnos puede que no sea divisible por el tamano de los cursos)
  
  for i in range(num_cursos):
    model+=(lpSum(rest_1[j][i]*ninas[j] for j in range(tot_alum))>=ninas_por_curso, 'ninas por curso '+str(i))
  
  #print('\nRestricciones de cantidad maxima por curso original')
  #(tiene que haber num_cursos_orig*num_cursos restricciones)
  #Esta restriccion es para hacer que todos los cursos tengan cantidades similares de alumnos de distintos cursos originales (no son iguales porque el total de alumnos puede que no sea divisible por el tamano de los cursos)
  
  for c in range(1,num_cursos_orig+1):
    for i in range(num_cursos):
      en_co=[ 1 if x ==c else 0 for x in curso_orig]
      model+=(lpSum(rest_1[j][i]*en_co[j] for j in range(tot_alum))>=max_mismo_curs_orig, 'curso orig '+str(c)+' en curso '+str(i))
  
  
  #print('\nRestricciones de alumnos incompatibles')
  #Tiene que haber n_cursos restricciones por cada grupo de incompatibles
  #Esta restriccion es para que no esten en el mismo cursos los alumnos incompatibles
  for c in range(n_incompat):
    for i in range(num_cursos):
      incomp=[1 if x==incompat_tp[c][0] or x==incompat_tp[c][1] else 0 for x in id]
      model+=(lpSum(rest_1[j][i]*incomp[j] for j in range(tot_alum))<=1, 'incompatibles '+str(c)+' en el curso '+str(i))
  
  #print('\nRestricciones de estar por lo menos con un escogido')
  
  for i in range(tot_alum):
    escog=[1 if x in escogidos_orig[i] else 0 for x in id]
    for k in range(num_cursos):
      model+=(lpSum(rest_1[j][k]*escog[j]  for j in range(tot_alum))>=rest_1[i][k], 'escogidos alumno'+str(i)+' en el curso '+str(k))
  
  
  #model.solve()
  model.solve(PULP_CBC_CMD())
  
  status =  LpStatus[model.status]
  
  print('Status de la optimizacion: ' + status)
  
  if status=='Optimal':
    print('👌')
  else:
    print('☹️')
  
  #print("Total Cost:", model.objective.value())
  
  
  DF_result0=pd.DataFrame()
  
  # Decision Variables
  
  for v in model.variables():
      try:
        if v.value()>0:
          #print(v.name,"=", v.value())
          df_aux={'nombre':v.name, 'valor':v.value()}
          DF_result0=DF_result0.append(df_aux, ignore_index=True)
      except:
          print("error couldnt find value")
  
  DF_result0['curso']=DF_result0['nombre'].str.slice( start=6, stop=7)
  DF_result0['id0']=DF_result0['nombre'].str.slice( start=-4)
  DF_result0[['id1', 'id']]=DF_result0['id0'].str.split('_', expand=True)
  DF_result0=DF_result0.drop(columns=['id0', 'id1', 'nombre'])
  DF_result0[['curso', 'id']]=DF_result0[['curso', 'id']].astype('int64')
  
  DF_result0=DF_result0.merge(datos_entrada, on='id', how='outer')
  
  DF_result0=DF_result0.sort_values(by=['curso', 'id'])
  DF_result0=DF_result0.drop(['valor'], axis=1)
  return(DF_result0)

# Evaluacion del resultado
#==============================

def eval_asig(df_result0):
  import re
  '''
  Se evalua el resultado que se obtuvo con la aplicación del ppl
  El DataFrame debe tener los nombres correctos
  '''
  print('''
  Evaluacion de los resultados
  ____________________________
  
        ''')
  
  DF_result=df_result0.copy()
  
  tot_alum=len(DF_result)
  num_cursos_orig=len(DF_result['curso_orig'].drop_duplicates())
  num_cursos=len(DF_result['curso'].drop_duplicates())
  tam_cursos=int((tot_alum//num_cursos))
  max_mismo_curs_orig=int((tot_alum)//(num_cursos*num_cursos_orig))
  ninas_por_curso=int(DF_result['nina'].sum()//num_cursos)
  num_escogidos=len([x for x in [re.findall(r"escogido", x ) for x in list(DF_result.columns)] if x!=[]])
  
  print("En cuantos cursos esta cada alumnos (duplicados)")
  
  print(DF_result['id'].duplicated().value_counts())
  
  if len(DF_result[DF_result['id'].duplicated(keep=False)])==0:
    print('👌')
  else:
    print('☹️')
    print(DF_result[DF_result['id'].duplicated(keep=False)])
  
  print('\n')
  print('Cantidad de alumnos por curso')
  print(DF_result.groupby('curso')['id'].count())
  if DF_result.groupby('curso')['id'].count().min()>=(tam_cursos):
    print('👌')
  else:
    print('☹️')
  print('\n')
  print('Cantidad de niñas por curso')
  print(DF_result.groupby('curso')['nina'].sum())
  
  if DF_result.groupby('curso')['nina'].sum().min()>=ninas_por_curso:
    print('👌')
  else:
    print('☹️')
  print('\n')
  print('Cantidad de curso original por curso')
  print(pd.crosstab(DF_result['curso'],DF_result['curso_orig']))
  if (pd.crosstab(DF_result['curso'],DF_result['curso_orig']).min().min())>=(max_mismo_curs_orig):
    print('👌')
  else:
    print('☹️')
  
  print('\n')
  print('Evaluacion de incompatibles')
  eval_incomp=DF_result[~(DF_result['incompatible'].isna())][['id', 'curso', 'incompatible']]
  
  eval_incomp['incompatible']=eval_incomp['incompatible'].astype('int64')
  
  eval_incomp_aux=eval_incomp[['id', 'curso']]
  eval_incomp_aux.columns=['incompatible', 'curso_incomp']
  
  eval_incomp=eval_incomp.merge(eval_incomp_aux, on='incompatible', how='left')
  eval_incomp['prob_incomp']=eval_incomp['curso']==eval_incomp['curso_incomp']
  
  print(eval_incomp['prob_incomp'].value_counts())
  print(eval_incomp)
  
  if len(eval_incomp[eval_incomp['prob_incomp']])==0:
    print('👌')
  else:
    print('☹️')
  
  print('\n')
  print('Evaluacion de escogidos')
  
  DF_escogidos=DF_result[['id', 'curso']+['escogido_'+str(x) for x in range(1,num_escogidos+1)]]
  #print(DF_escogidos.head())
  
  curso_esc=DF_result[['id', 'curso']]
  DF_escogidos['eval_esc']=False
  
  for e in range(1,num_escogidos+1):
    curso_esc.columns=['escogido_'+str(e), 'curso_e'+str(e)]
    #print(curso_esc.head())
    DF_escogidos=DF_escogidos.merge(curso_esc, on='escogido_'+str(e), how='left')
    #DF_escogidos['eval_esc']=np.where(DF_escogidos['curso']==DF_escogidos['curso_e'+str(e)], True, DF_escogidos['eval_esc'])
    DF_escogidos.loc[DF_escogidos['curso']==DF_escogidos['curso_e'+str(e)], 'eval_esc']=True
  
  print(DF_escogidos['eval_esc'].value_counts())
  
  if len(DF_escogidos[DF_escogidos['eval_esc']])==len(DF_escogidos):
    print('👌')
  else:
    print('☹️')
    print(DF_escogidos[~DF_escogidos['eval_esc']])
  
  
  DF_result=DF_result.merge(DF_escogidos[['id', 'eval_esc']], on='id', how='left')

  if len(DF_result[DF_result['id'].duplicated(keep=False)])==0 and DF_result.groupby('curso')['id'].count().min()>=(tam_cursos) and DF_result.groupby('curso')['nina'].sum().min()>=ninas_por_curso and (pd.crosstab(DF_result['curso'],DF_result['curso_orig']).min().min())>=(max_mismo_curs_orig) and len(eval_incomp[eval_incomp['prob_incomp']])==0 and len(DF_escogidos[DF_escogidos['eval_esc']])==len(DF_escogidos):
    print('\nSe cumplieron todas las restricciones!!! 👏')
  else:
    print('\nEste resultado es sub optimo ☹️')

  
# Utilizacion de las funciones

datos_entrada=gen_alum_sim(108, por_incomp=0.05)

result=asign_alum(datos_entrada)

eval_asig(result)


list(result.columns)
