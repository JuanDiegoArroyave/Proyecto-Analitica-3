----procesamientos---

---crear tabla con usuarios con más de 100 peliculas vistas y menos de 2,100

drop table if exists ratings_3 ;

CREATE TABLE ratings_3 AS 
WITH usuarios AS (
    SELECT 
        userId, 
        COUNT(*) AS cnt_rat
    FROM ratings_2
    GROUP BY userId
    HAVING cnt_rat > 100 AND cnt_rat <= 2100
),
peliculas AS (
    SELECT 
        movieId,
        COUNT(*) AS cnt_rat
    FROM ratings_2
    GROUP BY movieId
    HAVING cnt_rat >= 30
)
SELECT 
    a.userId,
    a.movieId,
    a.rating,
    a.timestamp,
    a.date
FROM ratings_2 a 
INNER JOIN usuarios b ON a.userId = b.userId  
INNER JOIN peliculas c ON a.movieId = c.movieId;  