# Riesgo de Crédito

Aplicación de Streamlit que predice si un solicitante va a pagar o a incumplir su crédito. Usa el Credit Risk Dataset de Kaggle y modelos entrenados en el notebook del proyecto.

La pestaña de predicción recibe los datos del solicitante en la barra lateral, deja escoger entre 8 modelos y muestra el resultado con la probabilidad de incumplir. El modelo recomendado es Gradient Boosting. La pestaña de dashboard resume los 32,409 créditos del dataset limpio.

## Estructura

```
riesgo-credito/
├── app.py
├── requirements.txt
├── data/
│   └── credit_risk_dataset.csv
└── modelos/
    └── modelos entrenados, escalador y columnas en formato .pkl
```

## Uso en local

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
