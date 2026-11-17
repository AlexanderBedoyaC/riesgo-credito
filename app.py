import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de Crédito")

carpeta = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def cargar_modelos():
    escalador = joblib.load(os.path.join(carpeta, "modelos", "escalador.pkl"))
    columnas = joblib.load(os.path.join(carpeta, "modelos", "columnas.pkl"))
    orden_grado = joblib.load(os.path.join(carpeta, "modelos", "orden_grado.pkl"))
    tasa_real = joblib.load(os.path.join(carpeta, "modelos", "tasa_real.pkl"))
    umbral = joblib.load(os.path.join(carpeta, "modelos", "umbral.pkl"))
    archivos = joblib.load(os.path.join(carpeta, "modelos", "archivos_modelos.pkl"))
    modelos = {}
    for nombre in archivos:
        modelos[nombre] = joblib.load(os.path.join(carpeta, "modelos", archivos[nombre]))
    return escalador, columnas, orden_grado, tasa_real, umbral, modelos


@st.cache_data
def cargar_datos():
    df = pd.read_csv(os.path.join(carpeta, "data", "credit_risk_dataset.csv"))
    df = df.drop_duplicates()
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]
    df = df[(df["person_emp_length"].isna()) | ((df["person_emp_length"] <= 60) & (df["person_emp_length"] <= df["person_age"] - 14))]
    df["estado"] = df["loan_status"].map({0: "Pagó", 1: "Incumplió"})
    return df


escalador, columnas, orden_grado, tasa_real, umbral, modelos = cargar_modelos()
df = cargar_datos()

# ---------------- Barra lateral ----------------
st.sidebar.header("Datos del solicitante")
edad = st.sidebar.slider("Edad", 18, 100, 30)
ingreso = st.sidebar.slider("Ingreso anual en dólares", 4000, 500000, 50000, step=1000)
empleo = st.sidebar.slider("Años trabajando", 0, 60, 3)
vivienda = st.sidebar.selectbox("Situación de vivienda", ["RENT", "OWN", "MORTGAGE", "OTHER"])
incumplio_antes = st.sidebar.selectbox("¿Tiene incumplimientos anteriores?", ["N", "Y"])

st.sidebar.header("Datos del préstamo")
proposito = st.sidebar.selectbox("Propósito del préstamo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                                                            "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
grado = st.sidebar.selectbox("Calificación del préstamo", ["A", "B", "C", "D", "E", "F", "G"])
monto = st.sidebar.slider("Monto solicitado en dólares", 500, 35000, 10000, step=500)
tasa = st.sidebar.slider("Tasa de interés anual en %", 5.0, 30.0, 11.0, step=0.1)

st.sidebar.header("Modelo")
opciones = ["Boosting", "Bagging", "Árbol de Decisión", "Votación", "SVM", "KNN", "Regresión Logística", "Naive Bayes"]
nombre_modelo = st.sidebar.selectbox("Seleccionar modelo", opciones)
st.sidebar.caption("Boosting es el modelo final del notebook. Usa un umbral de " + str(umbral) + " para detectar al menos 8 de cada 10 clientes que incumplen.")

predecir = st.sidebar.button("Predecir")

# ---------------- Pestañas ----------------
st.title("Predicción del Riesgo de Crédito")

tab1, tab2 = st.tabs(["Predicción", "Dashboard"])

with tab1:
    st.write("Esta aplicación predice si un solicitante va a pagar o a incumplir su crédito, según sus datos "
             "personales, los datos del préstamo y el modelo seleccionado. Los datos se llenan en la barra de la izquierda.")

    porcentaje = round(monto / ingreso, 2)

    datos = pd.DataFrame({
        "person_age": [edad],
        "person_income": [ingreso],
        "person_home_ownership": [vivienda],
        "person_emp_length": [empleo],
        "loan_intent": [proposito],
        "loan_grade": [grado],
        "loan_amnt": [monto],
        "loan_int_rate": [tasa],
        "loan_percent_income": [porcentaje],
        "cb_person_default_on_file": [incumplio_antes]
    })

    st.subheader("Datos ingresados")
    st.dataframe(datos, hide_index=True)

    if empleo > edad - 14:
        st.warning("Los años trabajando no pueden ser mayores a la edad menos 14.")

    if predecir:
        entrada = datos.copy()
        entrada["loan_grade"] = entrada["loan_grade"].map(orden_grado)
        entrada["cb_person_default_on_file"] = entrada["cb_person_default_on_file"].map({"Y": 1, "N": 0})
        entrada = pd.get_dummies(entrada, columns=["person_home_ownership", "loan_intent"], dtype=int)
        entrada = entrada.reindex(columns=columnas, fill_value=0)
        entrada = escalador.transform(entrada)

        modelo = modelos[nombre_modelo]
        if nombre_modelo == "Boosting":
            limite = umbral
            prediccion = int(modelo.predict_proba(entrada)[0][1] >= umbral)
        else:
            limite = 0.5
            prediccion = modelo.predict(entrada)[0]

        st.subheader("Resultado")
        st.write("Modelo usado:", nombre_modelo)

        if prediccion == 1:
            st.error("La predicción es: el cliente va a INCUMPLIR")
        else:
            st.success("La predicción es: el cliente va a PAGAR")

        if hasattr(modelo, "predict_proba"):
            probabilidad = modelo.predict_proba(entrada)[0][1]
            if nombre_modelo == "KNN":
                st.metric("Probabilidad de incumplir", str(round(probabilidad * 100, 1)) + " %")
                st.progress(float(probabilidad))
            else:
                ajustada = (probabilidad * tasa_real) / (probabilidad * tasa_real + (1 - probabilidad) * (1 - tasa_real))
                limite_ajustado = (limite * tasa_real) / (limite * tasa_real + (1 - limite) * (1 - tasa_real))
                st.metric("Probabilidad real de incumplir", str(round(ajustada * 100, 1)) + " %")
                st.progress(float(ajustada))
                st.caption("El modelo se entrenó dando el mismo peso a los clientes que pagan y a los que incumplen, "
                           "por lo que su probabilidad directa fue de " + str(round(probabilidad * 100, 1)) + " %. "
                           "La probabilidad real se ajusta con la tasa de incumplimiento de los datos, que es de "
                           + str(round(tasa_real * 100, 1)) + " %. Con este modelo se marca como riesgoso a quien pase de "
                           + str(round(limite_ajustado * 100, 1)) + " % de probabilidad real.")
        else:
            st.info("Este modelo no calcula probabilidades, solo la clase.")
    else:
        st.info("Llene los datos en la barra lateral y oprima Predecir.")

with tab2:
    st.write("Análisis de los " + f"{len(df):,}" + " créditos del dataset después de quitar duplicados y registros con errores. "
             "Los filtros cambian todas las gráficas de esta pestaña.")

    filtro_grado = st.multiselect("Calificación", sorted(df["loan_grade"].unique()), default=sorted(df["loan_grade"].unique()))
    filtro_vivienda = st.multiselect("Vivienda", sorted(df["person_home_ownership"].unique()), default=sorted(df["person_home_ownership"].unique()))
    filtro_proposito = st.multiselect("Propósito", sorted(df["loan_intent"].unique()), default=sorted(df["loan_intent"].unique()))

    filtrado = df[df["loan_grade"].isin(filtro_grado) & df["person_home_ownership"].isin(filtro_vivienda) & df["loan_intent"].isin(filtro_proposito)]

    if len(filtrado) == 0:
        st.warning("No hay créditos con esos filtros.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Créditos", f"{len(filtrado):,}")
        k2.metric("Tasa de incumplimiento", f"{filtrado['loan_status'].mean() * 100:.1f} %")
        k3.metric("Tasa de interés promedio", f"{filtrado['loan_int_rate'].mean():.2f} %")
        k4, k5 = st.columns(2)
        k4.metric("Ingreso mediano", f"$ {filtrado['person_income'].median():,.0f}")
        k5.metric("Monto promedio", f"$ {filtrado['loan_amnt'].mean():,.0f}")

        st.subheader("Incumplimiento por categoría")

        st.write("**Tasa de incumplimiento por calificación**")
        tabla = filtrado.groupby("loan_grade")["loan_status"].mean() * 100
        st.bar_chart(tabla, x_label="Calificación", y_label="% que incumplió")
        st.caption("Calificación con más incumplimiento: " + tabla.idxmax() + " con " + f"{tabla.max():.1f}" + " %.")

        st.write("**Tasa de incumplimiento por situación de vivienda**")
        tabla = filtrado.groupby("person_home_ownership")["loan_status"].mean() * 100
        st.bar_chart(tabla, x_label="Vivienda", y_label="% que incumplió")
        st.caption("Vivienda con más incumplimiento: " + tabla.idxmax() + " con " + f"{tabla.max():.1f}" + " %.")

        st.write("**Tasa de incumplimiento por propósito del préstamo**")
        tabla = (filtrado.groupby("loan_intent")["loan_status"].mean() * 100).sort_values()
        st.bar_chart(tabla, x_label="% que incumplió", y_label="Propósito", horizontal=True)
        st.caption("Propósito con más incumplimiento: " + tabla.idxmax() + " con " + f"{tabla.max():.1f}" + " %.")

        st.write("**Tasa de incumplimiento según incumplimientos anteriores**")
        tabla = filtrado.groupby("cb_person_default_on_file")["loan_status"].mean() * 100
        tabla.index = tabla.index.map({"N": "Sin antecedentes", "Y": "Con antecedentes"})
        st.bar_chart(tabla, x_label="Historial en central de riesgo", y_label="% que incumplió")

        st.subheader("Distribución de las variables numéricas")
        st.write("Cantidad de clientes en cada rango, separando los que pagaron de los que incumplieron.")

        st.write("**Edad**")
        tabla = filtrado.copy()
        tabla["rango"] = (tabla["person_age"] // 5) * 5
        tabla = tabla[tabla["rango"] <= 70]
        st.bar_chart(pd.crosstab(tabla["rango"], tabla["estado"]), x_label="Edad en años", y_label="Clientes")

        st.write("**Ingreso anual hasta 200,000 dólares**")
        tabla = filtrado[filtrado["person_income"] <= 200000].copy()
        tabla["rango"] = (tabla["person_income"] // 10000) * 10000
        st.bar_chart(pd.crosstab(tabla["rango"], tabla["estado"]), x_label="Ingreso en dólares", y_label="Clientes")

        st.write("**Monto del préstamo**")
        tabla = filtrado.copy()
        tabla["rango"] = (tabla["loan_amnt"] // 2500) * 2500
        st.bar_chart(pd.crosstab(tabla["rango"], tabla["estado"]), x_label="Monto en dólares", y_label="Clientes")

        st.write("**Proporción del ingreso que representa el préstamo**")
        tabla = filtrado.copy()
        tabla["rango"] = (tabla["loan_percent_income"] * 20).astype(int) / 20
        st.bar_chart(pd.crosstab(tabla["rango"], tabla["estado"]), x_label="Proporción del ingreso", y_label="Clientes")

        st.subheader("Comparación entre clientes que pagaron y que incumplieron")

        st.write("**Promedio de cada variable según el estado del crédito**")
        promedios = filtrado.groupby("estado")[["person_age", "person_income", "person_emp_length", "loan_amnt",
                                                "loan_int_rate", "loan_percent_income"]].mean().T
        promedios.index = ["Edad", "Ingreso anual", "Años trabajando", "Monto", "Tasa de interés", "Proporción del ingreso"]
        st.dataframe(promedios.round(2))

        st.write("**Tasa de interés promedio por calificación**")
        tabla = filtrado.groupby("loan_grade")["loan_int_rate"].mean()
        st.line_chart(tabla, x_label="Calificación", y_label="Tasa de interés %")

        st.write("**Correlación de cada variable numérica con el incumplimiento**")
        correlacion = filtrado[["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
                                "loan_percent_income", "cb_person_cred_hist_length", "loan_status"]].corr()["loan_status"]
        correlacion = correlacion.drop("loan_status").sort_values()
        st.bar_chart(correlacion, x_label="Correlación de Pearson", y_label="Variable", horizontal=True)

        with st.expander("Ver los datos filtrados"):
            st.dataframe(filtrado.drop(columns="estado"))
            st.download_button("Descargar CSV", filtrado.drop(columns="estado").to_csv(index=False), "datos_filtrados.csv")
