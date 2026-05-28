# PPVMD nonlinear feature classification pipeline

## 文件说明

- `ppvmd_ml_utils.py`  
  核心工具函数：合并、聚合、特征筛选、分类器、交叉验证、指标计算。

- `01_build_master_tables.py`  
  从 `results/<SubjectID>/main/` 读取每个受试者的 event/fixed 表，合并并聚合成受试者级表。

- `02_feature_screening.py`  
  单变量统计筛选，用于解释和候选特征排序。

- `03_run_classification.py`  
  多分类器对比实验，输出 ACC/SEN/SPE/F1/AUC。

- `04_plot_results.py`  
  绘制 event-guided vs fixed-window 的分类结果对比图。

## 推荐运行顺序

```bash
python 01_build_master_tables.py
python 02_feature_screening.py
python 03_run_classification.py
python 04_plot_results.py
```

## 重要原则

分类必须基于 subject-level features，不能直接把窗口当成独立样本。

event-guided 和 fixed-window 要分开建模，再比较二者分类性能。
