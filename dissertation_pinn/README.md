# PINN Dissertation Pipeline
## Physics-Informed Neural Networks for Cerebral Aneurysm Haemodynamic Risk Assessment
### University of Zimbabwe

## Pipeline Stages
| Stage | File | Description |
|-------|------|-------------|
| 1 | stage1_geometry.py | Geometry generation |
| 2 | stage2_pinn_caseA.py | Newtonian PINN — Hagen-Poiseuille validation |
| 3 | stage3_carreau_yasuda.py | Non-Newtonian Carreau-Yasuda PINN |
| 4 | stage4_curved_pipe.py | Curved pipe — Dean flow |
| 5 | stage5_aneurysm.py | Saccular aneurysm — primary clinical case |
| 6 | stage6_risk_assessment.py | Risk map and sensitivity analysis |

## Running on Kaggle (recommended — free T4 GPU)
```python
# In a Kaggle notebook cell:
!git clone https://github.com/MALVERN_MAMBO/dissertation-pinn.git
import sys, os
os.chdir('dissertation-pinn/stages')
sys.path.insert(0, '.')
%run stage1_geometry.py
```

## Requirements
- PyTorch
- NumPy
- SciPy
- Matplotlib
