# VisionAI

**VisionAI** est une application web basée sur Flask qui utilise l’intelligence artificielle (YOLOv8) pour :
- la détection d’objets dans des images, vidéos et flux webcam,
- le changement dynamique de l’arrière-plan,
- l’application de filtres d’image (noir et blanc, Sobel, etc.).

## 🚀 Fonctionnalités

- 🎯 Détection d’objets (YOLOv8) dans :
  - images uploadées
  - vidéos
  - webcam en temps réel
- 🎨 Application de filtres :
  - RGB
  - Niveaux de gris
  - Sobel (bords)
- 🧼 Suppression ou modification de l’arrière-plan avec `rembg`
- 💻 Interface web conviviale avec rendu dynamique des résultats
- 📥 Téléchargement du résultat traité

## 🧠 Technologies utilisées

- Python, Flask
- OpenCV
- YOLOv8 (`ultralytics`)
- cvzone
- rembg
- moviepy
- HTML/CSS (via Jinja2)

## 📁 Structure du projet
```bash
VisionAI/
│
├── static/
│ ├── uploads/ # Fichiers uploadés
│ └── output/ # Résultats traités (images, vidéos)
│
├── templates/
│ ├── home.html
│ ├── index.html
│ ├── result.html
│ ├── webcam.html
│ ├── back.html
│ ├── index2.html
│ └── result2.html
│
├── PFE.PY # Point d’entrée de l’application Flask
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier
```


## ⚙️ Installation

### Prérequis
- Python 3.8+
- `pip`

### Étapes


# 1. Cloner le dépôt
```bash
git clone https://github.com/hamzasabouri/VisionAI.git
cd VisionAI
```

# 2. Créer un environnement virtuel (optionnel)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

# 3. Installer les dépendances
```bash
pip install -r requirements.txt
```
# 4. Lancer l'application
```bash
python PFE.PY
```
L'application sera disponible sur : 
```bash
http://localhost:5000
```
## 🧪 Routes principales

| Route                 | Méthode(s)     | Description                                        |
|----------------------|----------------|----------------------------------------------------|
| `/`                  | GET            | Page d’accueil                                     |
| `/detection_image`   | GET, POST      | Upload d’image ou vidéo pour la détection d’objets|
| `/detection_webcam`  | GET            | Détection en temps réel via la webcam              |
| `/changer_background`| GET, POST      | Suppression ou changement de l’arrière-plan        |
| `/filtrage_image`    | GET, POST      | Application de filtres (grayscale, Sobel, etc.)    |
| `/video`             | GET            | Flux vidéo webcam en MJPEG                         |
| `/download/<filename>`| GET           | Télécharger une image ou vidéo traitée             |
| `/static/output/<filename>` | GET     | Accès aux fichiers de sortie                       |
