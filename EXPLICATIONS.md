# Fil rouge du projet : réseau de neurones à une couche cachée

> Ce fichier sert à préparer la défense orale en 1:1.
> Pour chaque fichier : **ce qu'il fait**, **les maths derrière**, **les choix faits et pourquoi**, et **les questions possibles du prof**.
> On le complète au fur et à mesure qu'on code.

---

## 0. Vue d'ensemble

### L'objectif
Coder from scratch, uniquement avec NumPy, un réseau de neurones avec **une seule couche cachée**, en deux versions :
- `SimpleSLPClassifier` : prédit une classe (binaire ou multi-classe)
- `SimpleSLPRegressor` : prédit un nombre

Les deux suivent l'interface de sklearn (`fit`, `predict`, `score`), ce qui permet de les comparer directement à `MLPClassifier` et `MLPRegressor`.

### Le schéma du réseau

```
X  ──►  couche cachée  ──►  couche de sortie  ──►  prédiction
        z1 = X @ W1 + b1     z2 = a1 @ W2 + b2
        a1 = g(z1)           y_pred = sortie(z2)
```

- `g` = fonction d'activation cachée (relu, tanh ou logistic)
- `sortie` = sigmoid (binaire), softmax (multi-classe) ou identité (régression)

### La boucle d'entraînement (descente de gradient)
1. **Forward** : calculer la prédiction
2. **Loss** : mesurer l'erreur
3. **Backward** : calculer les gradients avec la règle de la chaîne
4. **Update** : `W = W - learning_rate * dW`
5. Recommencer jusqu'à `max_iter` (ou early stopping)

### Organisation du code

| Fichier | Rôle |
|---|---|
| `activations.py` | Fonctions d'activation et leurs dérivées |
| `slp_base.py` | Classe mère commune (paramètres, initialisation des poids) |
| `slp_classifier.py` | Réseau pour la classification |
| `slp_regressor.py` | Réseau pour la régression |
| `test_*.py` | Tests fournis par le prof |

---

## 1. `activations.py`

*À compléter.*

---

## 2. `slp_base.py`

*À compléter.*

---

## 3. `slp_regressor.py`

*À compléter.*

---

## 4. `slp_classifier.py`

*À compléter.*

---

## 5. Rapport et résultats

*À compléter.*

---

## 6. Questions probables à l'oral

*À compléter au fil du projet.*
