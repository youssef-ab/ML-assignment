
# ML assignment - OCR
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
## Objectif
Dans ce projet, on vise à établir un algorithme IA permettant la reconnaissance des objets dans les documents. Il existe plusieurs modèles OCR permettant la détetion du texte avec une bonne précision. Mais actuellement, après quelques évolutions de modèles de documents, certain d'eux comprennent des éléments non textuels qui ont une valeur informative comme le code QR et code bar. Habituellement, ces deux éléments sont détectés par des pipelines différents de ceux qui se concentrent sur les éléments textuels. Mais dans le but de la parallélisation, on va explorer une approche différente ici : détectons tout en même temps.


## Approche utilisée
  
La détection d'objets est une tâche d'intelligence artificielle qui vise à détecter des objets dans des images. Yolo V5 est l'un des meilleurs modèles disponibles pour la détection d'objets actuellement. L'avantage de ce réseau neuronal profond est qu'il est très facile de le re-entrainer sur notre propre ensemble de données. Cet algorithme ne regarde l'image qu'une seule fois, en ce sens qu'il n'a besoin que d'un seul passage de propagation directe à travers le réseau pour faire des prédictions. Après une non-max suppression, il produit les objets détectés ainsi que leurs boîtes englobantes.
Par conséquent ce model est idéale pour notre problématique. En effet, on va modifier la dernière couche du réseau pour qu'il ne compte que 3 classes d'objets.
### Détails du modèle
#### Entrée et sortie
**Entrée** : L'entrée est un batch d'images, et chaque image a la taille (m, 608, 608, 3)
**Sortie** : La sortie est une liste de boîtes englobantes avec les classes reconnues. Chaque boîte englobante est représentée par 6 nombres $(p_c, b_x, b_y, b_h, b_w, c)$ comme expliqué ci-dessous. Si vous étendez $c$ en un vecteur à 80 dimensions, chaque boîte englobante est alors représentée par 85 nombres.
- $p_c$ : Probabilité qu'un objet soit présent dans une boîte englobante.
- $b_x, b_y$ :  Cordonnées du centre de la boite englobante.
- $b_h, b_w$ :  Hauteur et largeur de la boite englobante.
- $c$ : classe de l'objet détecté
#### Anchor Boxes 
- Les anchor Boxes sont choisies en explorant les données d'entraînement pour choisir des rapports hauteur/largeur raisonnables qui représentent les différentes classes.
- La dimension pour les anchor boxes est l'avant-dernière dimension dans l'encodage : $(m, n_H,n_W,ancres,classes)$.
- L'architecture YOLO est : IMAGE (m, 640, 640, 3) -> DEEP CNN -> ENCODAGE (m, n_H,n_W, n_anchors, 3).

### Implémentation
L'entraînement de ce réseau nécessite une puissance de calcul. C'est pour  cela, j'ai utilisé Google  colab  pour implémenter le modèle.  
Premièrement, j'ai installé certaines bibliothèques requises par le projet  (torch  1.5.1, torchvision 0.6.1, numpy 1.17, PyYAML 5.3.1, cocoapi.git), on va aussi installer Apex de NVIDIA pour accélérer l'apprentissage de notre modèle.  
#### Ensemble de données
YOLO  v5  nécessite que le jeu de données soit au format darknet. Voici un aperçu de ce à quoi cela ressemble :  
- Un fichier  txt  avec étiquettes par image  
- Une ligne par objet  
- Chaque ligne contient : class_index  bbox_x_center  bbox_y_center  bbox_width  bbox_height  
- Les coordonnées des boîtes doivent être normalisées entre 0 et 1.
Donc, j'ai codé une fonction permettant de créer l'ensemble de données compatible avec notre modèle à partir des données  fournies (fonction  create_dataset()). Cette fonction nous a permis de générer un ensemble de données d'entraînement  (90%)  et une de validation  (10%).

#### Configuration du projet YOLO v5
YOLO v5 utilise PyTorch, mais tout est abstrait. On a besoin du projet lui-même (ainsi que des dépendances requises).
D'abord, on  a cloné le  repo  GitHub  et vérifié un commit  spécifique  (pour assurer la reproductibilité). On a aussi besoin de deux fichiers de configuration. Un pour l'ensemble de données et un pour le modèle  (conf.yaml,  yolov5x.yaml). Ils permettent de changer le nombre de classes à 3 et de préciser les noms des classes et le chemin vers les données.

#### Entraînement
Concernant l'entraînement, j'ai utilisé le plus grand modèle YOLOv5x (89M paramètres), qui est aussi le plus précis.
Pour entraîner un modèle sur un ensemble de données personnalisé, j'ai appelé le script **train.py**. J'ai fait passer quelques paramètres :
- img 640 - redimensionne les images à 640x640 pixels
- batch 8 - 8 images par lot
- epochs 30 - s'entraîner pendant 30 epochs
- data ./data/conf.yaml - chemin vers la configuration de l'ensemble de données
- cfg ./models/yolov5x.yaml - configuration du modèle
- name yolov5x_exo2 - nom de notre modèle
- cache - cache les images du jeu de données pour un entraînement plus rapide
L'entraînement a pris environ 1h30 minutes. Le meilleur modèle est enregistré dans le fichier weights/best_yolov5x_exo2.pt.
####  Evaluation
Le projet contient une fonction plot_results() permettant d'évaluer les performances de notre modèle : 
On constate que la précision moyenne (mAP) s'améliore tout au long de la formation. Le modèle pourrait bénéficier d'un entraînement supplémentaire, mais ça va prendre beaucoup plus de temps.
#### Prédiction
On a choisis 50 images de l'ensemble de validation et on les a déplacé vers inference/images pour voir comment notre modèle se comporte sur celles-ci.
On a utilisé le script detect.py pour tester notre modèle sur les images. Voici les paramètres que nous utilisons :
- weights weights/best_yolov5x_exo2.pt - point de contrôle du modèle
- img 640 - redimensionne les images à 640x640 px
- conf 0.4 - prend en compte les prédictions avec une confiance de 0.4 ou plus élevée
- source ./inference/images/ - chemin vers les images

## Exécution
Pour reproduire les résultats que j'ai obtenu, télécharger l'ensemble des images d'entraînement  dans un dossier "samples" et les  labels  dans le répertoire  principal  "content". Ensuite, vous pouvez  exécuter  le  notebook  sur votre compte  colab  sans aucun souci. Vous pouvez sauter l'étape d'entraînement et passer directement à l'évaluation en téléchargeant le modèle entraîné sur le dossier "content/yolov5/weights/".
## Références
-   [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
-    [YOLOv5 Train on Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
-  [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)
