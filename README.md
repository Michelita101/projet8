# 🚗 Projet 8 — Segmentation d’images pour voiture autonome

Ce projet vise à développer un modèle de segmentation d’images embarqué dans un système de vision pour véhicule autonome. Il comprend :

- L'entraînement d’un modèle de deep learning sur le dataset **Cityscapes** (8 classes principales).
- Le déploiement d'une **API FastAPI** pour la prédiction.
- Une application **Streamlit** de présentation.
- Un suivi **MLflow**.
- Une **note technique** (≈10 pages) et un **support de soutenance**.

## 🎯 Objectifs techniques

- Manipuler un dataset volumineux (images + masques annotés).
- Concevoir un modèle de segmentation efficace et léger (U-Net, DeepLabv3+, etc.).
- Implémenter un générateur de données avec data augmentation.
- Évaluer avec des métriques robustes :
  - `pixel_accuracy`
  - `IoU` (Intersection over Union)
  - `mIoU` (mean IoU)
  - `Dice coefficient` (optionnel)
- Exposer le modèle via une API REST.
- Déployer une app web simple pour tester l’API.
- Livrer une démonstration fluide et documentée.

## 🛠️ Installation de l’environnement

```bash
conda activate seg-auto

## 📁 Arborescence projet (simplifiée)

PROJET8/
├── data/
│   ├── raw/           # Données brutes Cityscapes
│   └── processed/     # Données prétraitées (resize, masques encodés)
├── notebooks/         # Jupyter notebooks
├── src/
│   ├── data/          # Gestion data (loader, generator)
│   ├── models/        # Architectures modèles
│   └── utils/         # Fonctions communes
├── api/               # API FastAPI
├── app/               # Application Streamlit
├── reports/           # Note technique, figures
├── models/            # Modèles sauvegardés
├── scripts/           # Tâches automatisées (ex: prédiction batch)
├── environment.yml
├── README.md
└── .gitignore
