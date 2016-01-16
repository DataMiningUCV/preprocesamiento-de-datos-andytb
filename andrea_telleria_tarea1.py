# -*- coding: utf-8 -*-
"""
@author: atelleria
Andrea Tellería - CI:20614114
Tarea #1: Preprosesamiento
Minería de Datos
"""

#Cargando los paquetes necesarios
import numpy as np
import pandas as pd
from numpy import nan
from datetime import datetime
from sklearn.decomposition import PCA


#Guardando Data set como DataFrame con ayuda de pandas
df=pd.read_csv('data.csv', index_col=0, sep=',', quotechar='"', encoding='utf-8')

#Remplazando campos en string por valores numéricos donde:
#Si=1 y No=0
df.replace(u'No', 0, inplace=True)
df.replace(u'Si', 1, inplace=True)

#Enfermeria=1 y Bioanálisis=0
df.replace(u'Bioanálisis', 0, inplace=True)
df.replace(u'Enfermería', 1, inplace=True)

#Sexo: Femenino=0 y Masculino=1
df.replace(u'Femenino', 0, inplace=True)
df.replace(u'Masculino', 1, inplace=True)

#Estado Civil: Soltero (a)=0, Casado (a)=1, Viudo (a)=2 y Unido (a)=3
df.replace(u'Soltero (a)', 0, inplace=True)
df.replace(u'Casado (a)', 1, inplace=True)
df.replace(u'Viudo (a)', 2, inplace=True)
df.replace(u'Unido (a)', 3, inplace=True)

#Semestre Actual: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10...
df.replace(u"1er sem.", 1, inplace=True)
df.replace(u"2do\xa0sem.", 2, inplace=True)
df.replace(u"3er sem.", 3, inplace=True)
df.replace(u"4to\xa0sem.", 4, inplace=True)
df.replace(u"5to\xa0sem.", 5, inplace=True)
df.replace(u"6to\xa0sem.", 6, inplace=True)
df.replace(u"7mo sem.", 7, inplace=True)
df.replace(u"8vo\xa0sem.", 8, inplace=True)
df.replace(u"9no\xa0sem.", 9, inplace=True)
df.replace(u"10mo\xa0sem.", 10, inplace=True)
#df[u'Semestre.que.cursa'].tolist()

#Materias inscritas: más de 10 = 10, ya que no se específico se decidió dejar 10
df.replace(u'más de 10', '10', inplace=True)

#Lugar de procedencia
df.replace(u'Municipio Sucre', '0', inplace=True)
df.replace(u'Guarenas - Guatire', '1', inplace=True)
df.replace(u'Municipio Libertador Caracas', '2', inplace=True)
df.replace(u'Aragua', '3', inplace=True)
df.replace(u'Valles del Tuy', '4', inplace=True)
df.replace(u'Altos Mirandinos', '5', inplace=True)
df.replace(u'Apures', '6', inplace=True)
df.replace(u'Municipio El Hatillo', '7', inplace=True)
df.replace(u'Municipio Chacao', '8', inplace=True)
df.replace(u'Táchira', '9', inplace=True)
df.replace(u'Vargas', '10', inplace=True)
df.replace(u'Táchira', '11', inplace=True)
df.replace(u'Monagas', '12', inplace=True)
df.replace(u'Apure', '6', inplace=True)
df.replace(u'Trujillo', '13', inplace=True)
df.replace(u'Bolívar', '14', inplace=True)
df.replace(u'Barinas', '15', inplace=True)
df.replace(u'Sucre', '16', inplace=True)
df.replace(u'Barlovento', '17', inplace=True)
df.replace(u'Anzoategui', '18', inplace=True)
df.replace(u'Mérida', '19', inplace=True)
df.replace(u'Delta Amacuro', '20', inplace=True)
df.replace(u'Lara', '21', inplace=True)
df.replace(u'Yaracuy', '22', inplace=True)
df.replace(u'Guárico', '23', inplace=True)
df.replace(u'Municipio Baruta', '24', inplace=True)
df.replace(u'Nueva Esparta', '25', inplace=True)
df.replace(u'Portuguesa', '9', inplace=True)

#Creando una función para manejar formato de fechas, si el formato no es conocido, deja None
def parsing_date(birth_date):
    birth_date=str(birth_date)
    birth_date.encode('ascii', 'ignore')
    for formats in ('%d-%m-%y', '%d %m %y', '%d/%m/%y', '%d/%m/%Y', '%d/%m/%Y ', '%d%m%y', '%d-%m-%Y', '%d %m %Y', '%d%m%Y'):
        try:
            d=datetime.strptime(birth_date, formats)
            if d > datetime.now():
                d=datetime(d.year - 100, d.month, d.day)
            return d
        except ValueError:
            pass

#Pasando la función al DataFrame
birth_dates = df[u'Fecha.de.Nacimiento..colocar.sólo.datos.numéricos.'].tolist()
final_dates = []
for s in birth_dates:
    s=s.encode('ascii', 'ignore')
    s=parsing_date(s)
    final_dates.append(s)

#En el caso de fechas faltantes: Se busca la edad y con esta se escoge e año. De día se deja 12 y mes 12
#para prevenir que sea mayor a lo que debería.
for d in final_dates:
    if (d==None):
        c=final_dates.index(d)
        age = df[u'Edad'].tolist()
        l=age[c].split()
        f=datetime.now()
        final_dates[c]=datetime(f.year - int(l[0]), 12, 12)

##Modificando la columna existente
df[u'Fecha.de.Nacimiento..colocar.sólo.datos.numéricos.'] = final_dates

#Tomando en cuenta ciertos problemas encontrados en varios individuos al escribir números usano comas (,)
#y/o otros oroblemas menosres, se pasa a la corrección de estos
#Creando función para verificar que el promedio de aprovados sea menor o igual a 1
def val_efficiency(value):
    if (value>1): return (value/10000)
    return value

#Usando la función definida
efficiency = df[u'Eficiencia'].tolist()
final_efficiency = []
for i in efficiency:
    i=val_efficiency(i)
    final_efficiency.append(i)

#Modificando la columna existente
df[u'Eficiencia'] = final_efficiency


#En el caso de números en los que aparecen caracteres no válidos para un tipo numérico (ej. bs, comas (,), etc)
#Defino la función, en caso de que se tenga al menos un digito
def val_numbers (n_list):
    for i in n_list:
        try:
            if not(i.isdigit()):
                c=n_list.index(i)
                i=i.replace('bs', '')
                i=i.replace(' ', '')
                while (i.count(',')>1): 
                    i=i.replace(',', '',1)
                i=i.replace(',', '.')
                n_list[c]=float(i)
        except: pass
        
#En caso de que no hayan digitos en la fila
def val_digits (n_list):
    for i in n_list:
        try:
            i=str(i)
            if i.isalpha():
                c=n_list.index(i)
                n_list[c]=0
        except: pass
    
#Se usa la función en aquellas columnas en que se requiera
#Aporte que brinda el responsable económico
ing=df[u'Aporte.mensual.que.le.brinda.su.responsable.económico'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Aporte.mensual.que.le.brinda.su.responsable.económico']=ing

#Familiares y amigos
ing=df[u'Aporte.mensual.que.recibe.de.familiares.o.amigos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Aporte.mensual.que.recibe.de.familiares.o.amigos']=ing

#Actividades a destajo
ing=df[u'Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas']=ing

#Alimentación
ing=df[u'Alimentación'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Alimentación']=ing

#Transporte
ing=df[u'Transporte.público'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Transporte.público']=ing

#Gastos médicos
ing=df[u'Gastos.médicos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.médicos']=ing

#Gastos odontológicos
ing=df[u'Gastos.odontológicos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.odontológicos']=ing

#Gastos personales
ing=df[u'Gastos.personales'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.personales']=ing

#Residencia
ing=df[u'Residencia.o.habitación.alquilada'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Residencia.o.habitación.alquilada']=ing

#Materiales de estudio
ing=df[u'Materiales.de.estudio'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Materiales.de.estudio']=ing

#Recreación
ing=df[u'Recreación'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Recreación']=ing

#Otros gastos
ing=df[u'Otros.gastos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Otros.gastos']=ing

#Ingreso de su Responsable económico
ing=df[u'Ingreso.mensual.de.su.responsable.económico'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Ingreso.mensual.de.su.responsable.económico']=ing

#Otros ingresos 
ing=df[u'Otros.ingresos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Otros.ingresos']=ing

#Vivienda
ing=df[u'Vivienda'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Vivienda']=ing

#Alimentación familiar
ing=df[u'Alimentación.1'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Alimentación.1']=ing

#Transporte familiar
ing=df[u'Transporte'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Transporte']=ing

#Gastos médicos familiares
ing=df[u'Gastos.médicos.1'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.médicos.1']=ing

#Gastos odontológicos familiares
ing=df[u'Gastos.odontológicos.'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.odontológicos.']=ing

#Gastos educativos
ing=df[u'Gastos.educativos'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Gastos.educativos']=ing

#Servicios públicos
ing=df[u'Servicios.públicos.de.agua..luz..teléfono.y.gas'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Servicios.públicos.de.agua..luz..teléfono.y.gas']=ing

#Condominio
ing=df[u'Condominio'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Condominio']=ing

#Otros gastos
ing=df[u'Otros.gastos.1'].tolist()
val_digits(ing)
val_numbers(ing)
df[u'Otros.gastos.1']=ing


#Función en el caso de promedios ponderados, no pueden ser mayor a veinte (20)
def mean_p(n_list):
    for i in n_list:
        try:
            c=n_list.index(i)
            i=float(i)
            while (i > 20.00): i=i/10
            n_list[c]=0
        except:            
            pass   
#Aplicando la función
mean=df[u'Promedio.ponderado.aprobado'].tolist()
mean_p(mean)
df[u'Promedio.ponderado.aprobado']=mean


#Manejando variables nan o vacias
#En el caso de la columna sobre el Aporte recibido de familiares, en caso de no recibir respuesta se decidió 
#la media de la columna, asumiendo que el valor de parte de familiares o amigos no es cero.
df[u'Aporte.mensual.que.recibe.de.familiares.o.amigos']=df[u'Aporte.mensual.que.recibe.de.familiares.o.amigos'].fillna(df[u'Aporte.mensual.que.recibe.de.familiares.o.amigos'].mean())

#En el caso de la columna sobre el Aporte recibido de responzable económico: dejar la variable con el valor
#de la media de la columna, asumiendo que el valor no puede ser cero ya que se dijo previamente que se poseía un responzable económico
df[u'Aporte.mensual.que.le.brinda.su.responsable.económico']=df[u'Aporte.mensual.que.le.brinda.su.responsable.económico'].fillna(df[u'Aporte.mensual.que.le.brinda.su.responsable.económico'].mean())

#En el caso de la columna sobre el ingresos por actividades destajo, en caso de no recibir respuesta se decidió 
#usar la media de la columna, asumiendo que este valor de parte de destajos
df[u'Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas']=df[u'Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas'].fillna(df[u'Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas'].mean())

#En el caso de la columna sobre otros ingresos se tomó la media de la columna, asumiendo que no puede ser cero (0)
df[u'Otros.ingresos']=df[u'Otros.ingresos'].fillna(df[u'Otros.ingresos'].astype(float).mean())

#En el caso de las columnas de egresos (ej. alimentación) se decidió usar la media en caso de no recibir respuesta
df[u'Alimentación']=df[u'Alimentación'].fillna(df[u'Alimentación'].mean())
df[u'Transporte.público']=df[u'Transporte.público'].fillna(df[u'Transporte.público'].mean())
df[u'Gastos.médicos']=df[u'Gastos.médicos'].fillna(df[u'Gastos.médicos'].mean())
df[u'Gastos.odontológicos']=df[u'Gastos.odontológicos'].fillna(df[u'Gastos.odontológicos'].mean())
df[u'Gastos.personales']=df[u'Gastos.personales'].fillna(df[u'Gastos.personales'].mean())
df[u'Recreación']=df[u'Recreación'].fillna(df[u'Recreación'].mean())
df[u'Otros.gastos']=df[u'Otros.gastos'].fillna(df[u'Otros.gastos'].mean())

#En el caso de ciertas columnas de egresos (ej. residencia) se decidió usar cero (0) en caso de no recibir respuesta
df[u'Residencia.o.habitación.alquilada']=df[u'Residencia.o.habitación.alquilada'].fillna(0)

#En el caso de las columnas de ingresos (ej. otros ingresos) y egresos (ej. otros egresos)se decidió usar \
#la media en caso de no recibir respuesta
df[u'Otros.ingresos']=df[u'Otros.ingresos'].fillna(df[u'Otros.ingresos'].astype(float).mean())
df[u'Gastos.médicos.1']=df[u'Gastos.médicos.1'].fillna(df[u'Gastos.médicos.1'].astype(float).mean())
df[u'Gastos.odontológicos.']=df[u'Gastos.odontológicos.'].fillna(df[u'Gastos.odontológicos.'].astype(float).mean())
df[u'Gastos.educativos']=df[u'Gastos.educativos'].fillna(df[u'Gastos.educativos'].mean())
df[u'Servicios.públicos.de.agua..luz..teléfono.y.gas']=df[u'Servicios.públicos.de.agua..luz..teléfono.y.gas'].fillna(df[u'Servicios.públicos.de.agua..luz..teléfono.y.gas'].astype(float).mean())
df[u'Condominio']=df[u'Condominio'].fillna(df[u'Condominio'].astype(float).mean())
df[u'Otros.gastos.1']=df[u'Otros.gastos.1'].fillna(df[u'Otros.gastos.1'].mean())

#En el caso de ciertos ingresos (ej. Vivienda) se decidió usar cero (0) en caso de no recibir respuesta
df[u'Vivienda']=df[u'Vivienda'].fillna(0)


#Eliminando columnas que no aportan información al estudio 23-24
df.drop(df.columns[[0, 3, 8, 10, 11, 18, 21, 23, 24, 25, 26, 27, 28, 30, 32, 37, 47, 48, 52, 62, 63, 64]], axis=1, inplace=True)


#Buscando componentes principales del DataFrame resultante luego de limpiar la data y 
#eliminar errores en formatos y NaNs
#Sabiendo que el tipo datetime causa un error, sehace un drop a la columna df_aux
df_aux=df
df_aux.drop(df.columns[[1]], axis=1, inplace=True)
pca=PCA(n_components=5)
pca.fit(df_aux)


#Emitiendo la vista minable como un archivo csv
df.to_csv('out.csv', index=False, index_label=False, encoding='utf-8')