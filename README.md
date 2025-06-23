# PDAC  
Deep learning model for quantifying vascular invasion in pancreatic ductal adenocarcinoma

## Descriptions  
A deep learning framework for automatic segmentation and angle-based quantification of vascular invasion in pancreatic ductal adenocarcinoma.

## Highlights
- **Disease**: Pancreatic Ductal Adenocarcinoma (PDAC)
- **Goal**: Automated quantification of vascular invasion (VI)
- **Modalities**: Contrast-enhanced CT
- **Model**: Deep learning–based segmentation and angle computation
- **Applications**: Surgical decision-making, resectability evaluation, radiology support

## Functions  
- 加载增强CT动脉期/静脉期增强CT图像（NIfTI格式）
- 自动定位并提取以下结构的掩膜区域：
  - 动/静脉期肿瘤（Tumor）
  - 肠系膜上动脉（SMA）
  - 腹腔动脉（CA）
  - 肝总动脉（CHA）
  - 门静脉（PV）
  - 肠系膜上静脉（SMV）
- 计算肿瘤与动脉之间的三维/二维空间包绕角度
- 输出标准化评估表（如包绕角 >180° 提示手术困难）

## Releases  
2025.06.20 - v1.0  
- 支持NIfTI加载、角度计算、结构提取  
- 输出角度可视化截图和分析结果  

## About  
PDAC 项目由 The SJTU-Ruijin-UIH Institute for Medical Imaging Technology 开发，致力于建立临床实用的胰腺癌术前分析 AI 工具链，帮助医生实现肿瘤血管关系快速判断与切除可行性评估。

## Resources  
- [Readme](./README.md)  
- [Code](./src)  
- [Data (protected)](./data)  
- [Outputs](./output)

## Languages  
Python  
100.0%

---

© 2025 SJTU-Ruijin-UIH Institute
