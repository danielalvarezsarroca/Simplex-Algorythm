# Simplex Algorithm (Phase I–II) in Python

## Description
This project implements the **Simplex method** with **Phase 1** (finding a feasible solution) and **Phase 2** (optimization) in Python. The code solves linear programming problems in standard form using reduced costs and **Bland’s Rule** to avoid cycling. It also includes efficient updates of the basis inverse using elementary operations (matrix `E`).

The code is organized into:
1) **Helper functions** (z, reduced costs, feasible direction, θ, updates).
2) **Simplex algorithm** (initialization of Phase 1 & Phase 2 + main loop).
3) **Execution** (input reading and running the algorithm).

## Requirements
- Python 3.9+  
- `numpy`
- Custom module `lectura2.py` with the function `obtener_datos_optimizacion(...)`
- Data file `problemas.txt`

Install dependencies:
```bash
pip install numpy
