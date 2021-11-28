# NLP-IMDb


El proyecto tiene como finalidad desarrollar un modelo de procesamiento de lenguaje natural (NLP) que permita utilizar los algoritmos de aprendizaje automático sobre, en este caso, 50 mil críticas provenientes de Internet Movie Database IMDb. A partir de ello, creamos un predictor que permita distinguir críticas positivas y negativas.

La iniciativa para la elección de esta temática fue haber realizado, previamente, nuestro proyecto de Data Analytics sobre un dataset de Kaggle de la misma industria; y nos propusimos como objetivo hacer un análisis integral sobre la industria cinematográfica y recomendaciones de IMDb. Ergo, planteamos como objetivos de nuestro proyecto final lograr un mayor entendimiento de lo que compete el modelo NLP en Python, para poner en práctica estos conocimientos en futuros análisis; y que el modelo desarrollado para el dataset de IMDb permita que, al ingresar una nueva crítica, automáticamente el código pueda discernir qué valor (positivo o negativo) tendrá la misma.

<br>

### Los sprints involucrados en el proyecto son los siguientes:
1) Lectura del dataset de IMDb.
2) Limpieza de datos: 2a) Eliminación de marcadores HTML, emoticones irrelevantes y 2b) Utilización de stopwords para eliminar palabras vacías.
3) Armado de Nube de Palabras para ver cuáles son las más utilizadas en las reviews.
4) Obtención de palabras raíz mediante el algoritmo de Porter.
5) Creación y entrenamiento del modelo: 5a) Método LogisticRegression y 5b) Método RandomForestClassifier

<br>

### Las librerías utilizadas son:
1) numpy
2) wordcloud
3) matplotlib
4) re
5) nltk
6) sklearn
