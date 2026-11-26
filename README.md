# Pain Trajectories and Repeat MRI Findings in Acute and Chronic Temporomandibular Disorders: A Longitudinal Study

This repository contains the analysis code and interactive three-dimensional visualizations supporting the manuscript:

> **Pain Trajectories and Repeat MRI Findings in Acute and Chronic Temporomandibular Disorders: A Longitudinal Study**  
> Yeon-Hee Lee\*,†, Seonggwang Jeon†, Tae-Seok Kim, Yung-Kyun Noh, and Sang-woo Lee\*  
> *(†Co-first authors; \*Corresponding authors)*

## Interactive 3D Visualizations

The interactive versions of the exploratory three-dimensional visualizations (Supplementary Figures S5 and S6) are hosted via GitHub Pages:

- **Supplementary Figure S5a — Correlation Network (Initial VAS):**  
  [3D_Net_IVAS.html](https://seonggwangjeon.github.io/Clinical-Correlates-of-Pain-Intensity-and-Longitudinal-Pain-Change-in-Acute-and-Chronic-Temporomandi/3D_Net_IVAS.html)
- **Supplementary Figure S5b — Correlation Network (Final VAS):**  
  [3D_Net_FVAS.html](https://seonggwangjeon.github.io/Clinical-Correlates-of-Pain-Intensity-and-Longitudinal-Pain-Change-in-Acute-and-Chronic-Temporomandi/3D_Net_FVAS.html)
- **Supplementary Figure S6a — 3D Pain Distribution (Initial VAS):**  
  [3D_Space_IVAS.html](https://seonggwangjeon.github.io/Clinical-Correlates-of-Pain-Intensity-and-Longitudinal-Pain-Change-in-Acute-and-Chronic-Temporomandi/3D_Space_IVAS.html)
- **Supplementary Figure S6b — 3D Pain Distribution (Final VAS):**  
  [3D_Space_FVAS.html](https://seonggwangjeon.github.io/Clinical-Correlates-of-Pain-Intensity-and-Longitudinal-Pain-Change-in-Acute-and-Chronic-Temporomandi/3D_Space_FVAS.html)

## Repository Structure

```text
.
├── Code/
│   ├── common.py                    # Shared data loading and preprocessing utilities
│   ├── Table1.py                    # Baseline demographic, clinical, and MRI characteristics (Table 1)
│   ├── Table2.py                    # Longitudinal pain, clinical, and MRI changes by chronicity (Table 2)
│   ├── Table3.py                    # Initial pain correlates and chronicity interactions (Table 3)
│   ├── Table4.py                    # Final pain regression, random forest, and CV evaluation (Table 4)
│   ├── MRI_change_analysis.py       # Direction-based, individual-finding, and count-based MRI change analyses
│   ├── Figure1.py – Figure5.py      # Main manuscript figures (Figures 1–5)
│   ├── Supplementary_Figure_S1.py   # Count-based MRI progression and pain change (Supplementary Figure S1)
│   ├── Supplementary_Figure_S3.py   # Core random forest SHAP analysis (Supplementary Figure S3)
│   ├── Supplementary_Table_S1.py    # Nonparametric and bootstrap sensitivity analyses (Supplementary Table S1)
│   └── generate_supp_figures.py     # Supplementary Figures S2, S4, S5, S6 and 3D HTML visualizations
├── 3D_Net_IVAS.html                 # Interactive 3D correlation network (Initial VAS)
├── 3D_Net_FVAS.html                 # Interactive 3D correlation network (Final VAS)
├── 3D_Space_IVAS.html               # Interactive 3D scatter plot (Initial VAS)
├── 3D_Space_FVAS.html               # Interactive 3D scatter plot (Final VAS)
├── requirements.txt                 # Python 3.11.7 package versions used for verified analyses
├── LICENSE
└── README.md
