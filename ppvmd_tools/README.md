# PPVMD Tools 整理版

这是对原始工具函数文件的分类整理版本。

## 文件结构

- `ppvmd_denoising.py`  
  生理保护型预处理降噪主流程。

- `ppvmd_validation.py`  
  自适应参数推荐、半合成数据构造、去噪评价、消融实验。

- `ppvmd_vmd.py`  
  生理约束型 VMD 模态评分、重构、Mode 消融和 K-alpha 参数搜索。

## 推荐导入

```python
from ppvmd_tools.ppvmd_denoising import physiology_preserving_adaptive_denoise
from ppvmd_tools.ppvmd_validation import (
    recommend_adaptive_parameters,
    make_semisynthetic_pressure_dataset,
    evaluate_semisynthetic_result,
)
from ppvmd_tools.ppvmd_vmd import (
    physiology_constrained_vmd_reconstruction,
    summarize_physiology_vmd_result,
    search_best_vmd_parameters,
)
```

## 当前推荐参数

- VMD 输入：`Pressure_Clean`
- `min_keep_score = 0.18`
- 相空间重构导向主参数：`K = 6`, `alpha = 1500`
- 重构误差最优对照：`K = 3`, `alpha = 1500`
