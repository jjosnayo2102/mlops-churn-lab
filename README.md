# MLOps Churn Lab

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

* Validar los datos antes de entrenar.
* Garantizar que la API prediga lo mismo que el modelo evaluado.
* Elegir el modelo con evidencia y decidir con criterios de negocio.
* Promover a producción solo lo que pasa un quality gate.
* Detectar cuándo el modelo deja de funcionar y reentrenarlo.
* Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 26 passed
python src/validate_data.py
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── configs/churn.yaml              # configuración única (costos, contrato, quality gate)
├── src/
│   ├── common.py                   # carga de config y CSV crudo (keep_default_na=False)
│   ├── validate_data.py            # fase 1: quality gate de contrato de datos
│   ├── features.py                 # fase 2: ColumnTransformer + Pipeline
│   ├── data_prep.py                # fase 2: limpieza y split estratificado
│   ├── train.py                    # fase 3: baseline + 4 candidatos con CV y MLflow
│   └── evaluate_and_register.py    # fase 4: umbral por costo, segmentos, gate y aliases
├── tests/                          # tests de contrato, pipeline de features y API
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # ficha CRISP-ML(Q) y guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-validation → data-prep → train → evaluate
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Docker Hub, drift, reentrenamiento automático |

## Reto obligatorio · Cuarto modelo candidato (`gradient_boosting`)

Se agregó `GradientBoostingClassifier` como cuarto modelo candidato en `src/train.py` y se comparó en MLflow mediante validación cruzada estratificada (5 folds) junto al baseline y los otros tres candidatos:

| Modelo | F1 CV (media ± std) | ROC-AUC CV (media ± std) |
|---|---|---|
| `baseline` | `0.6629 ± 0.0075` | `0.5000 ± 0.0000` |
| `logistic` | `0.6666 ± 0.0454` | `0.7315 ± 0.0408` |
| `random_forest` | `0.6900 ± 0.0426` | `0.7326 ± 0.0421` |
| `xgboost` | `0.6829 ± 0.0656` | `0.7100 ± 0.0627` |
| `gradient_boosting` | `0.6476 ± 0.0761` | `0.6858 ± 0.0604` |

**¿Cambia el modelo elegido?**
**No.** `random_forest` sigue obteniendo el mayor `cv_f1_mean` (`0.6900 ± 0.0426`) y el mayor `cv_roc_auc_mean` (`0.7326 ± 0.0421`) con la menor varianza entre folds. En cambio, `gradient_boosting` obtuvo un F1 CV inferior (`0.6476 ± 0.0761`) y un ROC-AUC CV de `0.6858 ± 0.0604` (incluso por debajo del baseline en F1). Por lo tanto, `random_forest` se mantiene como el modelo ganador y fue promovido a `@champion` tras superar el quality gate en test (`F1 = 0.7013`, umbral `0.3`).