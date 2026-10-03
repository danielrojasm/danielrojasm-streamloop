# StreamLoop — Reporte de ajuste del modelo de churn

Notebook: [`streamloop_tuning.ipynb`](streamloop_tuning.ipynb) · Resultados en bruto: [`results.json`](results.json)

## Resumen

Un `RandomForestClassifier` con hiperparámetros por defecto detectaba solo **48 %** de los clientes que cancelan. Tras una búsqueda aleatoria amplia seguida de una grilla focalizada, optimizando **F2**, el modelo ajustado detecta **80 %**: en el test set se escapan **75** clientes en vez de **196**. A cambio, se envían más ofertas de retención innecesarias (288 falsos positivos frente a 111).

## Proceso

1. **Limpieza mínima:** se eliminó `customerID` y `TotalCharges` se convirtió a número (11 valores en blanco, de clientes con `tenure = 0`, pasan a NaN).
2. **Split 80/20 estratificado** (`random_state=42`) antes de cualquier modelado.
3. **Pipeline:** `ColumnTransformer` (imputación por mediana + `StandardScaler` en las numéricas, `OneHotEncoder` en las categóricas) → `RandomForestClassifier`. Todo el preprocesamiento vive dentro del pipeline, así que la imputación y el encoding se ajustan solo con los datos de entrenamiento de cada fold.
4. **Línea base** con hiperparámetros por defecto → **uso #1 del test set**.
5. **`RandomizedSearchCV`**: 60 combinaciones sobre `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` y `class_weight`, con `StratifiedKFold(5)` y `n_jobs=-1`, solo sobre `X_train`.
6. **`GridSearchCV`**: 96 combinaciones alrededor de la región ganadora, mismo CV, solo sobre `X_train`, con `refit=True` por defecto.
7. Revisión de `cv_results_` (promedio, std, peor y mejor fold) → elección → **uso #2 del test set**.

## Elección de la métrica: F2

- **Por qué no accuracy:** con un 27 % de churn, predecir "nadie cancela" ya da un 73 % de accuracy. La línea base tenía buena accuracy (0.78) y aun así perdía a más de la mitad de los clientes que cancelan.
- **Por qué no recall a secas:** se maximiza de forma trivial marcando a todos como churn, y la campaña de retención se volvería inútil.
- **F2** (`make_scorer(fbeta_score, beta=2)`) le da al recall **4 veces más peso** que a la precisión (β² = 4). Refleja el negocio: un cliente perdido sin detectar cuesta mucho más que un descuento ofrecido de más, pero no ignora del todo los falsos positivos.

## Lo que mostró la búsqueda aleatoria

- `class_weight` es el factor decisivo: los 10 mejores candidatos usan pesos balanceados y los 10 peores son todos `class_weight=None`.
- Árboles poco profundos (`max_depth` 4–6) y `min_samples_split` alto ganan; `max_features` en `sqrt` o `0.3`.
- `n_estimators` apenas influye.

Grilla resultante: `class_weight=balanced`, `n_estimators=400`, `max_depth ∈ {5,6,7,8}`, `max_features ∈ {sqrt, 0.3}`, `min_samples_leaf ∈ {1,3,6,10}`, `min_samples_split ∈ {10,20,30}`.

Mejor F2 en CV: búsqueda aleatoria **0.7272** → grilla **0.7292**.

## Estabilidad y elección del modelo final

| Candidato (grilla) | F2 medio en CV | Std entre folds | Peor fold | Mejor fold |
|---|---|---|---|---|
| **#1 depth 6, sqrt, leaf 3, split 30** (elegido) | **0.7292** | 0.0144 | **0.7057** | 0.7499 |
| #2 depth 6, sqrt, leaf 6, split 30 | 0.7285 | 0.0155 | 0.7035 | 0.7510 |
| #3 depth 6, sqrt, leaf 6, split 20 | 0.7284 | 0.0173 | 0.7048 | 0.7531 |
| Más estable del empate: depth 7, sqrt, leaf 10, split 30 | 0.7226 | **0.0128** | 0.7027 | — |

- El ruido entre folds (std ≈ 0.014, unos 0.04 de distancia entre el mejor y el peor fold) es **mucho mayor** que la diferencia entre los primeros candidatos (el top 10 está dentro de 0.003). **93 de los 96** candidatos caen dentro de 1 std del mejor, así que en la práctica están empatados.
- Por eso se comparó el de mayor promedio con la opción más estable. La "más estable" solo baja la std en 0.0016, pierde 0.0066 de promedio y además **tiene un peor fold más bajo** (0.7027 frente a 0.7057). Su menor varianza no da un piso más seguro.
- **Decisión:** el candidato de mayor promedio. Su std está por debajo de la del resto del top 3, tiene el mejor peor-fold del grupo y está bien regularizado (profundidad 6, `min_samples_split=30`). Es exactamente el `best_estimator_` que reentrenó `refit=True`; no se reentrenó nada a mano.

### Hiperparámetros finales

```python
RandomForestClassifier(
    class_weight="balanced",
    n_estimators=400,
    max_depth=6,
    max_features="sqrt",
    min_samples_leaf=3,
    min_samples_split=30,
    random_state=42,
)
```

## Línea base frente a modelo ajustado (test set, 1 409 clientes, 374 churners)

| Métrica | Línea base (por defecto) | Ajustado | Δ |
|---|---|---|---|
| **F2** (optimizada) | 0.4986 | **0.7177** | +0.219 |
| Recall | 0.4759 | **0.7995** | +0.324 |
| Precisión | 0.6159 | 0.5094 | −0.107 |
| F1 | 0.5370 | 0.6223 | +0.085 |
| ROC-AUC | 0.8176 | 0.8412 | +0.024 |
| Accuracy | 0.7821 | 0.7424 | −0.040 |
| Churners detectados (TP) | 178 | **299** | +121 |
| Churners perdidos (FN) | 196 | **75** | −121 |
| Ofertas innecesarias (FP) | 111 | 288 | +177 |

**Lectura:** el F2 en test (0.718) es coherente con el estimado de CV (0.729 ± 0.014), así que la búsqueda no sobreajustó la validación. La caída de accuracy y precisión es el trade-off esperado y buscado: se rescatan 121 clientes adicionales a cambio de 177 ofertas de retención de más, un intercambio favorable si perder un cliente cuesta más de unas 1.5 ofertas. La mejora del ROC-AUC (+0.024) indica además que el modelo ajustado ordena mejor a los clientes por riesgo, no solo que movió el umbral.

## Próximos pasos posibles

- Con costos reales de FN/FP, ajustar el umbral de decisión (`TunedThresholdClassifierCV`) o reemplazar F2 por un scorer de costo esperado.
- Usar `RepeatedStratifiedKFold` para reducir el ruido entre folds al comparar candidatos tan cercanos.
