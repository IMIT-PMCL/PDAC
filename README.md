# PAN-VIQ
Deep learning model for quantifying vascular invasion in pancreatic ductal adenocarcinoma

## Descriptions  
A deep learning framework for automatic segmentation and angle-based quantification of vascular invasion in pancreatic ductal adenocarcinoma.

## Installation

### Clone the repository
```bash
git clone https://github.com/IMIT-PMCL/PDAC.git
cd PDAC
```

## Highlights
- **Disease**: Pancreatic Ductal Adenocarcinoma (PDAC)
- **Goal**: Automated quantification of vascular invasion (VI)
- **Modalities**: Contrast-enhanced CT
- **Model**: Deep learning–based segmentation and angle computation
- **Applications**: Surgical decision-making, resectability evaluation, radiology support

## Functions  
- Load contrast-enhanced CT images in arterial and venous phases (NIfTI format)
- Automatically localize and extract masks for:
  - Tumor (arterial/venous phases)
  - Superior mesenteric artery (SMA)
  - Celiac artery (CA)
  - Common hepatic artery (CHA)
  - Portal vein (PV)
  - uperior mesenteric vein (SMV)
- Compute 3D spatial encasement angles between the tumor and vessels
- Output a standardized assessment report

## Releases  
2025.06.20 - v1.0  
- Supports NIfTI loading, segmentation, and angle computation
- Outputs analysis results  

## About  
Developed by the SJTU-Ruijin-UIH Institute for Medical Imaging Technology, the PAN-VIQ project aims to build clinically practical preoperative AI tools for PDAC, helping physicians rapidly assess tumor–vessel relationships and evaluate surgical resectability.

## Resources  


## Languages  
Python  
100.0%

---

© 2025 SJTU-Ruijin-UIH Institute
