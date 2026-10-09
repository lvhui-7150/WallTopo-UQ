# 数据集清单与用途

论文实验不需要把所有公开数据都下载一遍。建议按“主数据、拓扑/几何外部验证、跨域验证、尺度标定”四层准备。

## 0. 当前已经下载并整理

当前工程已经准备好：

| 数据集 | 图像-mask 对数 | 路径 |
|---|---:|---|
| Dais masonry | 240 | `data/raw/dais` |
| DeepCrack | 537 | `data/raw/deepcrack` |
| Crack500 | 471 | `data/raw/crack500` |
| CrackForest | 118 | `data/raw/crackforest` |

合并 manifest 位于 `data/raw/public_all.csv`，共 1366 对。SDNET2018 官方下载被 Cloudflare 阻断，未写入本地；它只有图像级标签，不是主体分割实验的必需项。详细来源和限制见 `data/raw/README.md`。

## 1. 必须下载的主数据

### Dais masonry crack dataset

- 用途：砌体墙面裂缝分割主实验。
- 标注：图像与像素 mask；公开 GitHub 版本含 240 对 224 x 224 图像和 mask。
- 下载：<https://github.com/dimitrisdais/crack_detection_CNN_masonry>
- 说明：公开子集规模偏小，最终论文最好获得完整 Dais 数据，或与自有墙体数据合并。

### Crack500

- 用途：细裂缝分割、拓扑断裂和几何量化的外部验证。
- 标注：像素级 mask。
- 下载：<https://github.com/fyangneil/pavement-crack-detection>

### DeepCrack

- 用途：细长裂缝、中心线连续性和 clDice 外部验证。
- 标注：像素级 mask；部分版本还包含裂缝中心线相关信息。
- 下载：<https://github.com/yhlleo/DeepCrack>

## 2. 跨域与复杂环境验证

### SDNET2018 wall subset

- 用途：混凝土墙/非裂缝背景预训练、误检分析和分类辅助任务。
- 标注：图像级二分类，不是裂缝 mask。
- 下载：<https://digitalcommons.usu.edu/all_datasets/48/>
- 说明：不能直接用于像素级主实验，除非有额外 mask。

### CrackForest / CrackTree200

- 用途：阴影、污染、低质量和复杂路面背景鲁棒性验证。
- 标注：像素级 mask。
- 下载：<https://github.com/cuilimeng/CrackForest-dataset>

### CrackSC 与 UAV-Crack500

- 用途：设备、尺度和复杂背景域偏移验证。
- 标注：像素级 mask。
- 获取方式：从对应论文作者公开仓库/补充材料获取。不同版本文件命名和许可不同，使用前先核对。

## 3. 尺度和专家等级数据

公开 mask 数据通常没有可靠的真实尺度，不能直接声称毫米宽度。要完成论文中的毫米级几何和 severity 实验，需要补充一小套自有数据：

- RGB 图像；
- 同一画面内有标尺、已知相机内参或摄影测量参考；
- 注明相机、镜头、工作距离、墙面材质和光照；
- 由多名专家按照统一量表给出裂缝等级；
- 同一墙面、同一序列不得跨 train/val/test。

没有尺度时，代码只报告 pixel width，并在报告中标记 `scale unavailable`。

## 4. 建议的实际下载组合

如果目标是尽快做出可发表的主体实验，建议先下载：

1. Dais masonry 数据；
2. Crack500；
3. DeepCrack；
4. SDNET2018 wall 子集；
5. CrackForest。

这五项可以覆盖主体分割、拓扑/几何验证、误检背景和跨域测试。CrackSC、UAV-Crack500 和带尺度自有数据放在第二轮扩展实验。

## 5. 统一 manifest

每个数据源先分别构建 manifest，再设置统一元数据：

```powershell
python scripts/build_manifest.py `
  --images data/raw/dais/images `
  --masks data/raw/dais/masks `
  --recursive `
  --output data/raw/dais/manifest.csv

python scripts/annotate_manifest.py `
  --manifest data/raw/dais/manifest.csv `
  --domain masonry `
  --material brick `
  --device camera `
  --group-prefix dais
```

对其他数据集重复上述步骤，然后合并：

```powershell
python scripts/merge_manifests.py `
  --manifests `
    data/raw/dais/manifest.csv `
    data/raw/crack500/manifest.csv `
    data/raw/deepcrack/manifest.csv `
    data/raw/crackforest/manifest.csv `
  --output data/real/all.csv
```

最终 CSV 至少包含：

```text
image,mask,severity,group,domain,material,device
```

`group` 用来防止同一面墙或连续帧泄漏；`domain` 用来做 leave-one-domain-out；`severity` 没有专家标签时留空。
