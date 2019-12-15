Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

![TSNE_Latent](TSNE_Latent.png)

## Tâches réalisées

### Import du projet existant
1. Fork du project via `github`

![fork](./fork.png)

2. Connexion Google Drive - Google Colab
```sh
❯ from google.colab import drive
❯ drive.mount('/content/drive')
```
3. Clone du projet forké `https://github.com/ccarlier92/TripletFace.git`
```sh
❯ !git clone https://github.com/ccarlier92/TripletFace.git
```
4. Dézippage du dataset 
```sh
❯ !unzip -F /content/drive/My\ Drive/Copie\ de\ dataset.zip
```
### Entraînement du modèle avec hyperparamètres modifiés

1.  Installation des librairies nécessaires
```sh
❯ !pip3 install -r /content/TripletFace/requirements.txt
```
2. Entrainement du modèle avec modification des hyper paramètres
```sh
❯ %cd TripletFace
❯ !python3 -m tripletface.train -m model -s ../dataset/ -e 5 -b 64 -i 240
```
--> Nombre d'epoch réduit à 5 car c'est suffisant et ça réduit le temps d'entraînement de manière significative
--> Augmentation du batch size à 64 (32 par défaut) offrant une visualisation plus propre
--> Augmentation de l'input size à 240 (224 par défaut) offrant également une visualisation plus propre, mais allongeant considérablement le temps d'entraînement. Je n'ai donc pas cherché à monter plus haut.

En sortie, on obtient les visualisations des 5 epoch (format PNG) ainsi que le fichier model.pt. Tout cela est disponible dans le dossier TripletFace/model

### Jit Compile
1. Déplacement des dossiers vers le drive pour enregistrement futur
```sh
❯ !mv /content/TripletFace /content/drive/My\ Drive/IA4IOT
```
2. Compilation du réseau pour réutilisation plus rapide par la suite
```sh
❯ %cd /content/drive/My Drive/IA4IOT/TripletFace
❯ import torch
❯ from tripletface.core.model import Encoder
❯ model = Encoder(64)
❯ weigths = torch.load( "/model/model.pt" )['model']
❯ model.load_state_dict( weigths )
❯ jit_model = torch.jit.trace( model,torch.rand(3, 3, 3, 3) )
❯ torch.jit.save( jit_model, "jitcompile.pt" )
❯ %cd ..
```
## References

* Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
* Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
* TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:
* Fork the Project
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
* Add script to generate Centroids and Thesholds using few face images from one person
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model
* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
* Send the github link by mail
