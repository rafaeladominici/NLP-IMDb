# NLP-IMDb

<br>

## Introducción
El proyecto tiene como finalidad desarrollar un modelo de procesamiento de lenguaje natural (NLP) que permita utilizar los algoritmos de aprendizaje automático sobre, en este caso, 50 mil críticas provenientes de Internet Movie Database IMDb. A partir de ello, creamos un predictor que permita distinguir críticas positivas y negativas.

La iniciativa para la elección de esta temática fue haber realizado, previamente, nuestro proyecto de Data Analytics sobre un dataset de Kaggle de la misma industria; es por esto que nos propusimos continuar con la misma temática y hacer un análisis integral sobre la industria cinematográfica y recomendaciones de IMDb. Ergo, planteamos como objetivos de nuestro proyecto final lograr un mayor entendimiento de lo que compete el modelo NLP en Python, para poner en práctica estos conocimientos en futuros análisis; y que el modelo desarrollado para el dataset de IMDb permita que, al ingresar una nueva crítica, automáticamente el código pueda discernir qué valor (positivo o negativo) tendrá la misma.

<br>

## Objetivos del Proyecto
Podemos resumir los objetivos del presente trabajo de la siguiente manera:

1) Aprender todo aquello que esté relacionado con el procesamiento del lenguaje natural (NLP) tanto en terminos teóricos (conceptos asosciados, teorías que fueron desarrollando esta disciplina, entre otros) como en términos practicos con las librerías e hiperparámetros asociados al mismo.
2) Aprender sobre la manipulación y limpieza del dataset obtenido en pos de hacer un modelo mas eficiente y robusto, para ello utilizaremos todas las herramientas aprendidas a lo largo del curso que sean útiles para la obtención del resultado buscado.
3) Desarrollar y buscar un modelo que provea los mejores resultados a la hora de clasificar la review de una película, buscando la manera de que dicho modelo pueda ser usado por un usuario de manera facil e intuitiva.

<br>

## Preguntas de Investigación
Como mencionamos anteriormente, en el presente trabajo se utilizará un dataset de IMDb que contiene 50000 reviews de diferentes películas y, para cada una de ellas, una clasificación que puede ser 0 (review negativa) o 1 (review positiva) basada en la puntuación que el usuario finalmente le otorgó a la película. Al encontrarnos ante este dataset y combinarlo con los objetivos planteado, nos surjen necesariamente una serie de preguntas que buscaremos responder a lo largo del trabajo para que dichos objetivos puedan ser cumplidos. De esta manera, las preguntas de investigación que surgen son:

¿Cuántas críticas negativas y positivas tiene el dataset? ¿Es un dataset equilibrado? En caso de que no, ¿Qué técnica sería mejor utilizar para compensar este desbalanceo?
En las reviews, ¿Cuáles son las palabras que más se repiten? ¿Las mismas son coherentes con el tipo de industria sobre la que estamos trabajando? ¿Hay elementos que no sean palabras y que puedan perjudicar al modelo? En caso de existir, ¿Qué podemos hacer para corregirlo?
Una vez que contemos con las review de manera correcta, ¿Cómo transformamos estas reviews en un formato que sea asimilable por un modelo? ¿Qué modelos supervisados de clasificación existen y cual usamos?
Por último, ¿Cómo funciona el modelo? ¿Qué métricas podemos usar para medir los resultados del mismo? ¿Clasifica bien una nueva review que le enviamos?

A través de las diferentes lineas de código iremos respondiendo estas preguntas que servirán de guia para definir los pasos a seguir a lo largo de todo el proyecto.

<br>

## Pasos del proyecto

El proyecto se realizará íntegramente en una notebook con código Python, a modo de guía, los pasos que se seguirán serán los siguiente:

1) Lectura del dataset de IMDb.
2) Exploración y Limpieza de datos: a) Análisis de variable a predecir. b) Analisis de reviews con Nube de Palabras. c) Eliminación de marcadores HTML, emoticones irrelevantes. d) Obtención de palabras raíz mediante el algoritmo de Porter. e) Utilización de stopwords para eliminar palabras vacías.
3) Creación y entrenamiento del modelo: a) Método LogisticRegression y b) Método RandomForestClassifier.
4) Evaluación del modelo.
5) Creación de Interfaz Gráfica para ingresar nuevas críticas y visualizar si las mismas corresponden a valoracioens positivas o negativas.

<br>

## Librerías utilizadas
A su vez, las librerías a utilizar son:

1) pandas: librería para trabajar con datos tabulares en Python, se basa en numpy y permite realizar una gran cantidad de operaciones.
2) wordcloud: librería que facilita la creación de nube de palabras, muy importante a los efectos del presente proyecto.
3) matplotlib y seaborn: librerias de visualización de datos que facilitan la comprensión de la información mostrandola gráficamente.
4) re: libreria para utilizar expresiones regulares de forma de manipular texto y realizar la limpieza del mismo.
5) nltk: librería muy utilizada en el NLP ya que ofrece métodos para trabajar con texto como eliminar palabras frecuentes y las raices de las palabras.
6) sklearn: librería que cuenta con diferentes algoritmos de machine learning para el desarrollo de modelos.
7) tkinter: librería que nos permite crear interfaces gráficas que ejecuten el código de manera mas amigable para el usuario.

<br>

## Desarrollo de pasos realizados en la Notebook
En este apartado buscaremos explicar paso a paso la notebook del proyecto final, de manera que se logre entender la lógica seguida y las lineas de código ejecutadas. Para eso iremos viendo cada una de las etapas descriptas previamente.

### Lectura del dataset de IMDb.
En este paso lo primero que se realizó fue la importación de la librería Pandas que, como se mencionó previamente, permite trabajar con datos tabulares y ofrece métodos muy utiles para la manipulación de tablas por lo que será de mucha utilidad a lo largo de todo el proyecto.

En segundo lugar, se crea una variable llamada df que almacenara nuestro dataset, para eso se hace una lectura con el método de Pandas llamada read_csv al cual le indicamos el nombre que tiene el archivo en nuestra computadora. Además, a modo preventivo, le pasamos como parametro encodigo="utf-8" para unificar los diferentes formatos particulares creados en zonas distintas ya que, si bien las reviews estan en inglés, puede tener caracteres que distintas zonas, de esta manera no nos estaremos perdiendo de ninguno.

### Exploración y Limpieza de datos
En esta etapa se buscó realizar una exploración para comprender el dataset con el que estamos trabajando, ver sus particularidades y en base a eso hacer la limpieza correspondiente para que quede optimizado de cara al futuro.

A los efectos de responder la primera pregunta de investigación planteada previamente, lo primero que realizamos fue un breve analisis de la varible a predecir, para ver si estamos en presencia de una dataset balanceado o desbalanceado y, en función de ello, si debíamos realizar algun tratamiento adicional. Para ello utilizamos la librería seaborn para graficar a traves de su método countplot ,que recibe la columna y el dataset como parámetro,la variable a predecir y obtuvimos un dataset totalmente balanceado (25000 reviews positivas y 25000 negativas). Esta información nos dió la certeza de que no se requeriran tecnicas como oversampling y subsampling para corregir un desbalanceo. En esta intancia nuevamente se utilizó Pandas.

Lo siguiente que se realizó fue una nube de palabras para poder visualizar a simple vista cuáles son las palabras que más se repiten en el total de las reviews y si las mismas tienen sentido con la industría con la que estamos trabajando. Para realizar esta nube de palabras, necesitamos de la librería especialmente creada para eso llamada wordcloud con sus métodos WordCloud, STOPWORDS, ImageColorGenerator (tal vez finalmente no se usen todos pero buscamos asegurarnos que los importamos desde un principio) y matplotlib. Una vez importadas las librerías, instanciamos el objeto Wordcloud almacenandolo en una variable; como parámetro le pasamos width, height y colormap para definir su tamaño y estilo. Esta librería sólo acepta como input un string, es decir, una cadena de texto, unimos con un join cada elemento de la lista (review) y procedimos a alamcenarlo en una variable. De esta manera, alcanzamos una gran cadena de texto con todas las variables. Luego con la clásula wordcloud.generate y pasándole la lista generada anteriormente, creamos la nube de palabras. Para graficarla, utilizamos el método figure e imshow de matplotlib. Podemos notar que lo que más se repite era "<br>" que son las letras utilizadas para los marcadores HTML y emoticones, por lo que fue necesario quitarlos porque podrían generar problemas en el modelo.

Gracias a lo detectado previamente en la nube de palabras en donde pudimos notar que gran cantidad de reviews cuentan con marcadores HTML y emoticones sin significado, realizamos una eliminación de los mismos. Para esta instancia, importamos la librería re util para la minupación de texto y creamos una funcion llamada limpieza que lo que hace es excluir todo aquello que no sea texto y todo aquello que sea un emoticón dejarlo al final de la review a partir de los métodos sub y findall. Después aplicamos la función sobre dos reviews de ejemplo que incluyeron lo necesario para corroborar que funciona adecuadamente y, una vez hecho esto, usamos el método de Pandas apply sobre la columna review del dataframe para que lo aplique a cada una de las filas de dicha columna.

Con la limpieza de las reviews volvimos a hacer la nube de palabras para corroborar que lo referido a los marcadores HTML fue eliminado y sólo nos quedaron palabras normalmente utilizadas en reviews de películas.

Lo último que realizamos para esta etapa del proyecto fue eliminar las stopwords (palabras sin significado como artículos, pronombres, preposiciones, etc) de las reviews para que el aprendizaje del modelo sea más eficiente. Para ello utilizamos la librería nltk y el método stopwords de la misma. Creamos una variable llamada stopwords en donde le indicamos que las palabras vacias que queremos son en el idioma inglés. En este sentido creamos una funcion llamada depuración en donde lo que hacemos es: a traves de nltk.tokenize.word_tokenize para convertir una cadena de texto (en este caso una review) en un lista en donde cada elemento de la lista es una palabra de la cadena de texto (básicamente lo que hacemos es tokenizar un texto), después con una List comprehension obtenemos una lista en donde solo se queda con las palabras (tokens) que no son stopwords (es un proceso similar al que hicimos en la función "limpieza" para crear la nube de palabras pero hecha de manera mas eficiente con una list comprehension). Una vez que contamos con la lista con todas las palabras que no son palabras vacías deshacemos la lista para que nos quede un string. Nuevamente con el metodo apply lo aplicamos a todas las reviews del dataset.

### Creación y entrenamiento de modelo
Una vez que contamos con el dataset totalmente optimizado, el siguiente paso fue la creación del modelo y su entrenamiento. Es en esta instancia en donde entra en juego la libreria sklearn de la cual usaremos: Pipeline,LogisticRegression, CountVectorizer, train_test_split, TfidTransformer

Una vez que tenemos importadas las librerías lo primero que hacemos es crear una variable llamada pipeline para instanciar la clase Pipeline para, de esta manera, gestionar el flujo de trabajo. Primero incluimos ('bow', CountVectorizer()) para que convierta la colección de reviews en una matriz de recuentos de tokens, es decir, toma todas las palabras existentes en el df (recordemos que ya estará optimizado) y, para cada reviews, cuenta cuantas veces aparece. En segundo lugar incluimos ('tfidf', TfidfTransformer()) que va a tomar el resultado de CountVectorizer y le va a asignar un peso relativo a cada palabra de acuerdo a la cantidad de veces que aparece en una misma review y la cantidad de veces que se repite en el resto de las review. En tercer lugar incluimos ('classifier', LogisticRegression(random_state = 42)) que es el modelo de clasificación que seleccionamos para este proyecto (tambien hicimos el mismo proceso con Random Forest y el resultado fue similar).

Almacenamos en una variable X las reviews y en una variable y los resultados, para luego dividir cada conjunto de datos en subconjuntos de prueba y entrenamiento a través del método train_test_split en donde indicamos que el 30% de los datos sea de prueba con test_size=0.30.

Ya con el conjunto de datos dividido, entrenamos el modelo con el método fit() guardándolo en una variable para despues realizar la predicción con el método predict() al cual le pasamos como parámetro el subconjunto de prueba (X_test).

### Evaluación del modelo
Para evaluar el modelo tomamos las métricas de evaluación mas comunes para los modelos de clasificación como el que vimos, siendo estas las clases de sklearn confusion_matrix y classification_report.

Dentro de la matriz de confusión vemos rapidamente que el modelo clasificó correctamente a la gran mayoría de las críticas. Esto se ve reflejado en todas las métricas que obtuvieron un resultado similar. A modo de ejemplo, el Accuracy del 90% nos indica que del total de datos de prueba el modelo acertó el 90% de las veces (13450/15000), a su vez del total de veces que el modelo indicó que la crítica era negativa acerto el 91% mientras que en las criticas positivas el acierto es de un 89%.

Consideramos que los resultados son lo suficientemente precisos para responder correctamente ante nuevas críticas. Lo que ocurre muchas veces en este tipo de trabajos es que la crítica puede resultar ambigua incluso para un ser humano, entonces en esos casos puede ser que el modelo fallé pero en aquellas críticas que son relativamente comprensibles sabemos que el modelo funcionará bien.

### Interfaz Gráfica
A los efectos de que el modelo sea utilizado de manera facil e intuitiva por cualquier usuario, se nos ocurrio crear una interfaz gráfica capaz de recibir una nueva crítica y otorgar inmediatamente una devolución respecto al tipo de critica, es decir, si es una critica positiva o negativa.

Para la creación de la interfaz gráfica decidimos utilizar la librería tkinter y con ella lo primero que hacemos es instanciar un objeto, luego con los metodos title() y geometry() le ponemos un titulo y un tamaño, luego incluimos Labels, cuadro de texto y un botón con los métodos Label(),Entry() y Button y con el método grid() los ubicamos dentro del espacio (Frame).

Lo más importante es cómo indicar que al apretar el botón se ejecute una acción, en este caso, la clasificación de la crítica. Lo antes dicho se realiza con el parámetro command al cual se le indica la función a ejecutar, esta función la llamamos clasificar_critica que basicamente toma lo que el usuario ingresa en el cuadro de texto (Entre) a traves del método get(), lo almacena en una variable y se hace una predicción sobre el modelo entrenado pasandole la variable en donde esta almacenada la crítica. El resultado luego se ve en un espacio guardado para ello.

<br>

## Conclusiónes Finales
