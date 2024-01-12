# workspace-unsup-text-classification
Prueba de concepto de herramienta de clasificación de texto no supervisada



# pasos

# 1. EDA
- Analisis exploratorio por palabras. ver cual es el conjunto de palabras mas comunes.
- sacar diferentes estadisticas por titulo de noticia y por contenido.

# 2. Solucion de clasificacion

## solucion general

1. Preprocesado y limpieza
2. análisis de clustering y topics
3. Generación de categorías generales
4. Construcción de un modelo en base a esas categorias generadas
5. Solución de extracción de entidades
6. construcción de la solucion via scripts y apificacion
7. documentacion


## 2.1. Preprocesado y sampleo.

Vamos a trabajar con una muestra pequeña del dataset para ir mas rapido. un 20-30% una ver procesado.

Necesitamos preprocesar para limpiar info tanto de titulos como d econtenido, peor rpincipalmente contenido. Aqui planteamos
- Limpiar con codigo
- probar a limpiar con llm

- importante: hay cadenas de texto con @ @ @ @ @ que parecen palabras censuradas u ocultas. Quizas este dataset es para rellenar antes que clasificar.

## 2.2. Clustering no supervisado

Técnicas de clustering clasicas:

- text clustering using nlp techniques https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95 
    - tfidf
    - word2vec & doc2vec
    - embeddings de llms
- otro similar https://medium.com/@evertongomede/clustering-text-in-natural-language-processing-unveiling-patterns-and-insights-8c3cd137b135
- info en sklearn https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
- clustering con gpt3 embeddings https://www.youtube.com/watch?v=ld3YbhoJz9w
- clustering con sentence transformers https://www.youtube.com/watch?v=OlhNZg4gOvA
- clustering con bert embeddings https://www.youtube.com/watch?v=Kc9gN_gODvQ

Aqui la idea es emplear estas tecnicas para sacar embeddings y poder hacer clustering, a ver que grupos salen.

Aqui nos vamos a quedar con 3 opciones:
- bert
- sentence transformers
- openai

Y luego haremos clustering: kmeans o hdbscan
- kmeans https://archive.is/F8yMQ
- dbscan https://dylancastillo.co/clustering-documents-with-openai-langchain-hdbscan/

## 2.3. Topic modeling

Generacion de topics con modelos preentrenados

- https://www.youtube.com/watch?v=5a5Dfft-rWc
- https://maartengr.github.io/BERTopic/index.html bertopic

La idea es generar topics para tener info junto a los clusterings e intentar agrupar de alguna forma

## 2.4. Pruebas con zero-shot clasiffication

emplear modelos preentrenados para hacer la clasificacion según tematica.

Podemos probar cada modelo para ver que resultado nos da y tener info adicional

- OpenAI + prompt engineering
- hugging face (en el curso de hugging face lo hace)
    - https://huggingface.co/Voicelab/vlt5-base-keywords?text=Lawyer+who+took+on+tobacco+industry+now+turning+to+opioids+fight extractor de keywords
    - https://huggingface.co/Yueh-Huan/news-category-classification-distilbert?text=Lawyer+who+took+on+tobacco+industry+now+turning+to+opioids+fight categorizador de noticias
    - https://huggingface.co/jonaskoenig/topic_classification_04?text=Lawyer+who+took+on+tobacco+industry+now+turning+to+opioids+fight otro
    - https://huggingface.co/cardiffnlp/tweet-topic-21-multi?text=Lawyer+who+took+on+tobacco+industry+now+turning+to+opioids+fight categorizador de tweets
    - https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all?text=Lawyer+who+took+on+tobacco+industry+now+turning+to+opioids+fight otro
    - API vanilla de hugging face para clasificar por tematica (hay que pasarle labels)


## 2.5. construccion solucion final de clasificación

La idea es que a partir de una conbinación de esas soluciones o la selección de alguna, podamos generar un conjunto de labels a emplear luego para etiquetar.

con esos labels podemos:

- construir un modelo supervisado a partir de otro preentrenado
- Emplear un zero-shot potente que permita indicar los labels


# 3. Solución de extracción de entidades






        - tecnicas medianamente clasicas: encoding con word2vec o embeddings o similar y luego clusterizacion
        - solucion planteada: Usar lbl2vec https://github.com/sebischair/lbl2vec
            - lbl2vec Necesita un grupo de keywords asociados a topics generales. habria que ver como hacerlo. en el paper es conocimiento humano. en este caso, podríamos sacarlos con algun modelo de hugging face que saque keywords o entidades.
            - Aqui hay otro ejemplo de extracciond e topics https://medium.com/@power.up1163/unsupervised-text-classification-with-topic-models-and-good-old-human-reasoning-da297bed7362 lo podemos usar para probar y además tiene visualizaciones chulas de conceptos



