# Fundamentos de Análisis y Cálculo Numérico — proyecto PyTorch

## Objetivo y alcance
Implementar y documentar las cinco primeras guías Udacity de PyTorch y adaptarlas a clasificación binaria de radiografías de fractura, en dos entregas de la Universidad del Valle. Materiales originales y datos se conservan íntegros. El trabajo no constituye un sistema clínico.

## Tecnologías y dependencias
Python 3.13; entorno local `.venv`; PyTorch, torchvision, Pillow, NumPy, pandas, matplotlib, scikit-learn, Jupyter y nbformat. Las versiones instaladas quedan registradas en `requirements.txt`. Los notebooks originales fueron escritos para PyTorch 1.1; las copias de trabajo corrigen diferencias de API y dimensiones para la instalación actual.

## Estructura real del proyecto

```text
Fundamentos-Calculo-Numerico/
    ├── data/
    │   └── raw/
    │       ├── Bone_Fracture_Binary_Classification/
    │       └── README.dataset.txt
    ├── docs/
    │   └── informes/
    │       └── borrador_informe.md
    ├── notebooks/
    │   ├── ejecutados/
    │   │   ├── Part 1 - Tensors in PyTorch - ejecutado.ipynb
    │   │   ├── Part 2 - Neural Networks in PyTorch - ejecutado.ipynb
    │   │   ├── Part 3 - Training Neural Networks - ejecutado.ipynb
    │   ├── fracturas/
    │   │   ├── Guia 2-3 - fracturas.ipynb
    │   └── originales/
    │       └── DL_PyTorch/
    ├── results/
    │   ├── figures/
    │   ├── metrics/
    │   │   ├── fracture_image_inventory.csv
    │   │   └── smoke_test_untrained_one_epoch.json
    │   └── models/
    ├── src/
    │   └── fracture_experiment.py
    ├── README.md
    └── requirements.txt
```

## Material y datos
- Original Udacity: `notebooks/originales/DL_PyTorch/` (cinco notebooks, `helper.py`, imágenes y README originales).
- Copias corregidas para MNIST/Fashion-MNIST: `notebooks/ejecutados/`.
- Adaptaciones independientes de fracturas: `notebooks/fracturas/`; utilidades compartidas: `src/fracture_experiment.py`.
- Dataset único: `data/raw/Bone_Fracture_Binary_Classification/{train,val,test}/{fractured,not fractured}`. No se duplicó. Conteos observados: train 4,606 + 4,640; val 337 + 492; test 238 + 268. Se verificaron los 10,581 archivos: 10,362 JPEG, 216 PNG y 3 GIF internos; 9,512 miden 224×224 y se observó un JPEG truncado que Pillow tolera explícitamente. Inventario completo en `results/metrics/fracture_image_inventory.csv`.
- Resultados, métricas e inventario: `results/metrics/`; figuras: `results/figures/`; modelos: `results/models/`.
- Informe técnico específico de entrega 1: `docs/informes/entrega1_informe_tecnico.md`.
- Informe maestro único: `docs/informes/borrador_informe.md`. El borrador de presentaciones se creará cuando las guías estén verificadas y haya resultados reales suficientes.

## Configuración en Windows, VS Code y Jupyter
Desde PowerShell en la raíz:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name fundamentos-calculo-numerico --display-name "Python (Fundamentos Calculo Numerico)"
python -m jupyter lab
```

En VS Code, abrir esta raíz, instalar/activar la extensión Jupyter si se requiere y seleccionar el kernel `fundamentos-calculo-numerico`. Ejecutar primero guías 1–3 MNIST; luego la adaptación 2–3 de fracturas. Para segunda entrega, seguir guías 4–5 Fashion-MNIST y finalmente la adaptación 4–5. El script auxiliar resuelve las rutas desde la raíz y no escribe dentro del dataset.

## Orden recomendado
1. `Part 1 - Tensors in PyTorch - ejecutado.ipynb`
2. `Part 2 - Neural Networks in PyTorch - ejecutado.ipynb`
3. `Part 3 - Training Neural Networks - ejecutado.ipynb`
4. `Guia 2-3 - fracturas.ipynb`
5. `Part 4 - Fashion-MNIST Exercise - ejecutado.ipynb`
6. `Part 5 - Inference and Validation - ejecutado.ipynb`
7. `Guia 4-5 - fracturas.ipynb`

Las guías 1–5 originales están disponibles solo para consulta. En la copia de guía 5, al no contener el material un split de validación Fashion-MNIST, se reserva determinísticamente parte del split oficial de prueba como validación didáctica y el resto queda para inferencia final; para experimentos formales debe descargarse/separarse train en entrenamiento/validación. En fracturas se usa `val` oficial y `test` solo al finalizar.

