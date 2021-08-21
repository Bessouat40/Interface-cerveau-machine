# Interface-cerveau-machine

Une interface cerveau machine désigne une liaison entre un cerveau et un ordinateur.
Dans notre cas, nous voulions détecter des mouvements de la langue, du bras et de la jambe à l'aide d'électroencéphalogrammes correspondant à l'activité du cerveau à un moment donné.

# Dispositif :

Nous disposions de données brutes recueillies à l'aide d'un casque à EEG (électroencéphalogramme). Pour faire fonctionner ce casque, il faut disposer des électrodes sur la tête de la personne dont on souhaite récupérer l'activité du cerveau.
Ce casque permet de mesurer l'activité électrique du cerveau.

# Traitement des données : 

Nous ne sélections que quelques électrodes, celles qui correspondaient aux endroits du cerveau qui commandaient la langue, les bras et les pieds.
Une fois récoltées, nous traitions les données en utilisant un filtrage par ondelettes.

# IA :

Ensuite nous avons choisi d'utiliser deux types de solutions :
- utiliser des réseaux de neurones pré-entrainés
- utiliser un réseau de neurones entrainé avec nos données (réseau de neurones à une entrée puis à trois entrées)

# Conclusion :

Finalement nous obtenons des premiers résultats satisfaisants avec le transfer learning et nos réseaux de neurones multimodaux.
