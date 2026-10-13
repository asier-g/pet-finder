# PetFinder: predicción de la velocidad de adopción

Predicción de la velocidad de adopción de mascotas (clasificación ordinal en 5 clases) con los datos de la competición [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) de Kaggle.

Proyecto de la asignatura *Advanced Machine Learning* (Ingeniería Informática + Ciencia de Datos e IA, Universidad de Deusto).

## Resultados

**QWK = 0,445 en test** con un *stacking* de cinco modelos de ensemble (CatBoost, LightGBM, XGBoost, Random Forest y Gradient Boosting) y una regresión logística como meta-modelo.

El test es una partición propia: el 33 % estratificado del `train.csv` de Kaggle (4.948 mascotas), que no se usó en ningún momento para entrenar ni para ajustar. Todos los modelos de la tabla se evaluaron sobre esa misma partición.

| Modelo (test) | F1-macro | QWK |
|---|---|---|
| **Stacking (5 modelos + regresión logística, C = 0,1)** | 0,346 | **0,445** |
| CatBoost (categóricas nativas) | 0,352 | 0,439 |
| Soft voting (pesos 1-2-5-2-1, más peso a CatBoost) | 0,380 | 0,434 |
| XGBoost + RandomOverSampler | 0,372 | 0,419 |
| Random Forest (`class_weight="balanced"`) | 0,350 | 0,402 |
| LightGBM + RandomOverSampler | 0,355 | 0,399 |
| Gradient Boosting | 0,320 | 0,394 |

- **Compromiso entre métricas.** El stacking obtiene el mejor QWK porque sus errores caen cerca de la clase real, pero nunca predice la clase minoritaria (clase 0: adoptada el mismo día, el 2,7 % de los datos). El voting y XGBoost aciertan algunos casos de esa clase y por eso tienen más F1-macro.
- **No es comparable con el leaderboard de Kaggle.** No evaluamos sobre el test oficial de la competición, sino sobre nuestra propia partición del conjunto de entrenamiento.

![Matriz de confusión del stacking en test](images/4_part/Final_Stacking_Classifier_Test_Predictions_Confusion_Matrix.png)

## Enfoque

- **Variables a partir de JSON.** Hicimos un ETL de las salidas de Google Vision (una por imagen) y de Google Natural Language (una por descripción). De Vision sacamos cuánto ocupa el animal en la foto, la confianza del recorte, el brillo y la probabilidad de "puppy", agregados por mascota. De Natural Language, la puntuación y la intensidad del sentimiento, y si la descripción menciona WhatsApp o una llamada. Los valores ausentes se imputan con valores fuera de rango y con indicadores (`has_name`, `has_sentiment_data`, …), porque que falte un dato también es información.
- **Codificación sin *data leakage*.** Comparamos seis combinaciones de One-Hot, Target y Leave-One-Out encoding. Nos quedamos con LOOE para todas las categóricas (con ruido gaussiano, `sigma`), que era la que menos sobreajustaba. El preprocesador va dentro del `Pipeline`, así que se ajusta solo con el train de cada fold de la validación cruzada.
- **Desbalanceo: 86 combinaciones.** Probamos 7 modelos de ensemble con 11 técnicas de remuestreo (SMOTE, ADASYN, NearMiss, SMOTEENN, …), más las variantes con `class_weight="balanced"` y CatBoost con categóricas nativas, todo con validación cruzada estratificada de 5 folds. Los pesos por clase funcionaron mejor que generar o eliminar muestras. El sobremuestreo superó al submuestreo, y SMOTEENN y NearMiss fueron las peores técnicas.
- **Optimización con Optuna.** Usamos TPE con 100 *trials* por modelo. Hay poda por fold (`MedianPruner`): un *trial* se aborta si su QWK parcial queda por debajo de la mediana. Cada estudio arranca con la mejor configuración conocida (*warm start*) y los rangos se refinaron en un segundo estudio. El `sigma` del LOOE se optimizó junto a los hiperparámetros del modelo y resultó ser de los más influyentes según el análisis de importancia (fANOVA).
- **Voting y stacking.** Elegimos los pesos del soft voting por validación cruzada. Para el stacking comparamos cinco meta-modelos, entre ellos una regresión Ridge redondeada para respetar el orden de las clases. Ganó la regresión logística, y su `C` se ajustó con `GridSearchCV`.

**Por qué QWK.** Las clases están ordenadas (0 = mismo día … 4 = sin adoptar a los 100 días). Confundir la clase 0 con la 1 no es tan grave como confundirla con la 4. El *Quadratic Weighted Kappa* penaliza cada error según el cuadrado de la distancia entre la clase real y la predicha, y además era la métrica oficial de la competición. Usamos F1-macro, MCC y balanced accuracy como métricas secundarias, para no perder de vista la clase minoritaria.

![QWK por modelo y técnica de desbalanceo](images/2_part/imbalance_performance_distribution.png)

## Estructura del repositorio

```
├── preprocessing/
│   ├── preprocessing.ipynb          # ETL de los JSON, limpieza e imputación → data/cleaned/cleaned_df_final.parquet
│   ├── preprocessor.py              # ColumnTransformers (StandardScaler + One-Hot / Target / LOOE)
│   └── initial_analysis/            # distribución de la variable objetivo
├── main/
│   └── main.ipynb                   # split train/test, baselines, comparación de encoders, evaluación final en test
├── imbalance_analysis_mitigation/
│   ├── imbalance_analysis_mitigation.ipynb   # 86 combinaciones modelo × técnica de desbalanceo
│   └── technique_results/           # métricas de validación cruzada (CSV)
├── hyperparameter_tuning/
│   ├── hyperparameter_tuning.ipynb  # Optuna, importancia de hiperparámetros, selección del meta-modelo
│   ├── optuna_studies/              # estudios de Optuna (SQLite)
│   └── optimized_results.csv
├── images/                          # figuras del informe, por partes
└── data/                            # datos de Kaggle y particiones generadas (ver abajo)
```

## Cómo reproducirlo

1. **Descarga los datos** de la [competición](https://www.kaggle.com/c/petfinder-adoption-prediction/data). Antes hay que aceptar sus normas en Kaggle.
2. **Colócalos en `data/`** con estos nombres:

   | Kaggle | Repositorio |
   |---|---|
   | `train/train.csv` | `data/data.csv` |
   | `train_metadata/` | `data/data_metadata/` |
   | `train_sentiment/` | `data/data_sentiment/` |
   | `train_images/` | `data/train_images/` |
   | `breed_labels.csv`, `color_labels.csv`, `state_labels.csv` | `data/` |

3. **Instala las dependencias.** El proyecto se ejecutó con Python 3.11 y 3.12.
   ```bash
   pip install -r requirements.txt
   ```
4. **Ejecuta los notebooks desde su propia carpeta**, porque usan rutas relativas (`../data/...`), en este orden:
   1. `preprocessing/preprocessing.ipynb`: genera `data/cleaned/cleaned_df_final.parquet`.
   2. `main/main.ipynb`, celda *Dataset splitting*: genera `data/cleaned/X_train.parquet`, `X_test`, `y_train` e `y_test` (33 % de test, estratificado, `random_state=42`). La celda está comentada para no rehacer el split por accidente.
   3. `imbalance_analysis_mitigation/imbalance_analysis_mitigation.ipynb`: unos 15 minutos con 14 núcleos.
   4. `hyperparameter_tuning/hyperparameter_tuning.ipynb`: por defecto carga los estudios guardados en `optuna_studies/`. Las funciones objetivo están comentadas; descoméntalas para relanzar la búsqueda.
   5. `main/main.ipynb`, resto del notebook: entrena los modelos finales con todo el train, los evalúa en test y guarda las matrices de confusión en `images/4_part/`.

`preprocessing/initial_analysis/initial_analysis.py` es un script exploratorio que usa la estructura original de Kaggle (`data/train/train.csv`), no la de la tabla anterior.

## Equipo

- **Asier** <!-- TODO: apellido + enlace a GitHub/LinkedIn -->: optimización de hiperparámetros con Optuna y análisis de importancia de hiperparámetros.
- **David** <!-- TODO: apellido + enlace a GitHub/LinkedIn -->: preprocesado y ETL de los JSON.
- **Iker** <!-- TODO: apellido + enlace a GitHub/LinkedIn -->: análisis y mitigación del desbalanceo.
