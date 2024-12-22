# Movies content based filtering

## Overview

This is a basic movie recommendation system for suggesting similar movies based on their descriptions, genres, and ratings. It leverages natural language processing techniques, including `TF-IDF` and `Sentence-BERT` embeddings, to generate vector representations of movie data. These embeddings are stored and managed enabling fast similarity computations using `FAISS`. it includes custom embedding generators, a flexible recommender class for handling recommendations, and utility functions for managing embeddings.The dataset used is available here : `https://www.kaggle.com/code/omeroruccelik/content-based-recommendation-systems/input`

## Example of output
For the movie `Norm of the North: King Sized Adventure (movie_index = 0)` using `TF-IDF` embeddings :

```
original Movie (Norm of the North: King Sized Adventure):
description: Before planning an awesome wedding for his grandfather, a polar bear king must take back a stolen artifact from an evil archaeologist first.
rating: TV-PG
genres: Children & Family Movies, Comedies

similar Movies:
title: Norm of the North: Keys to the Kingdom
description: When Norm the polar bear is framed for a crime he didn’t commit, his friends step up to help him clear his name.
rating: TV-PG
listed In: Children & Family Movies

title: Garuda Di Dadaku
description: A determined boy will do anything to become a great soccer player, despite the wishes of his grandfather.
rating: TV-PG
listed In: Children & Family Movies, Dramas, Sports Movies

title: Ottaal
description: Young orphan Kuttappayi goes to live with his grandfather in the country, but his new life takes a hard turn when his grandfather's health fails.
rating: TV-PG
listed In: Dramas, International Movies

title: DJ Cinderella
description: Fiercely independent and disillusioned with love, a teen DJ is determined to chart her own path, till a pop heartthrob falls for her awesome mix.
rating: TV-PG
listed In: Children & Family Movies, Comedies

title: Manje Bistre
description: While prepping for a family wedding, a young man falls for his sister’s friend, who has already been promised to someone else.
rating: TV-PG
listed In: Comedies, Dramas, International Movies
```

For the same movie `Norm of the North: King Sized Adventure (movie_index = 0)` using `Sentence-BERTF` embeddings :

```
original Movie (Norm of the North: King Sized Adventure):
description: Before planning an awesome wedding for his grandfather, a polar bear king must take back a stolen artifact from an evil archaeologist first.
rating: TV-PG
genres: Children & Family Movies, Comedies

similar Movies:
title: Norm of the North: Keys to the Kingdom
description: When Norm the polar bear is framed for a crime he didn’t commit, his friends step up to help him clear his name.
rating: TV-PG
listed In: Children & Family Movies

title: The Croods
description: When an earthquake obliterates their cave, an unworldly prehistoric family is forced to journey through unfamiliar terrain in search of a new home.
rating: PG
listed In: Children & Family Movies, Comedies

title: Open Season
description: After saving a deer from a hunter's clutches, a domesticated grizzly finds himself relocated to the wild – and unprepared for the real world.
rating: PG
listed In: Children & Family Movies, Comedies

title: A Family Affair
description: The filmmaker hunts for the missing puzzle pieces of his family history during a visit with a complex and controversial figure: his grandmother.
rating: TV-PG
listed In: Documentaries, International Movies

title: Pachamama
description: When a sacred statue is taken from his Andean village, a spirited boy who dreams of becoming a shaman goes on a brave mission to get it back.
rating: PG
listed In: Children & Family Movies
```

## Comparaison
The TF-IDF and Sentence-BERT models produce distinct sets of similar movies, reflecting their underlying methodologies. The TF-IDF model relies on term frequency and word relevance within the movie descriptions. As a result, its recommendations include movies with some shared thematic elements or keywords but may miss deeper contextual connections. For instance, "Garuda Di Dadaku" and "Ottaal," though sharing a TV-PG rating and familial themes, are less directly connected to the adventure-driven plot of the original movie. This illustrates TF-IDF's limitation in understanding nuanced relationships.

In contrast, Sentence-BERT leverages deep learning to encode semantic meaning, enabling it to capture richer contextual similarities. Its recommendations, such as "The Croods" and "Open Season," share both the adventure and comedic elements of the original movie, creating a more thematically cohesive list. But, this model may sometimes prioritize thematic alignment over textual overlap, as seen with the inclusion of "A Family Affair," a documentary that semantically connects through its familial focus but diverges in format.
