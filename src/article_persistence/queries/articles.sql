-- name: insert_article<!
-- Insert a new article and return the article record (excluding full_text)
INSERT INTO articles (
    url,
    title,
    section,
    published_date,
    fetched_at,
    full_text,
    news_source_id
)
VALUES (
    :url,
    :title,
    :section,
    :published_date,
    :fetched_at,
    :full_text,
    :news_source_id
)
RETURNING id, public_id, url, title, section, published_date, fetched_at, news_source_id;

-- name: get_article_by_public_id^
-- Retrieve an article by its public UUID (for API lookups)
SELECT id, public_id, url, title, section, published_date, fetched_at, full_text, news_source_id
FROM articles
WHERE public_id = :public_id;

-- name: get_article_detail_by_public_id^
-- Retrieve a single article with aggregated entities and classifications by public UUID.
WITH article_classifications AS (
    SELECT article_id,
           jsonb_agg(jsonb_build_object(
               'classifier_type', classifier_type,
               'confidence_score', confidence_score,
               'reasoning', reasoning
           )) AS classifications
    FROM classifications
    GROUP BY article_id
)
SELECT
    a.public_id, a.url, a.title, a.section, a.published_date,
    ns.id AS news_source_id,
    NULL::text AS snippet,
    a.full_text,
    COALESCE(array_agg(DISTINCT e.name) FILTER (WHERE e.id IS NOT NULL), '{}') AS entities,
    COALESCE(ac.classifications, '[]'::jsonb) AS classifications
FROM articles a
JOIN news_sources ns ON a.news_source_id = ns.id
LEFT JOIN article_classifications ac ON a.id = ac.article_id
LEFT JOIN article_entities ae ON a.id = ae.article_id
LEFT JOIN entities e ON ae.entity_id = e.id
WHERE a.public_id = :public_id
GROUP BY a.id, a.public_id, a.url, a.title, a.section, a.published_date,
         a.full_text, ns.id, ac.classifications;

-- name: get_existing_urls
-- Check which URLs from a list already exist in the database
-- Returns set of existing URLs for filtering
SELECT url
FROM articles
WHERE url = ANY(:urls::text[]);

-- name: get_related_articles_by_public_id
-- Find articles related to the given article by shared entities, across all
-- news sources. :limit controls the maximum number of results returned.
-- The JOIN on news_sources is only to populate news_source_id in the response;
-- it does NOT filter related articles to the same source as the target article.
WITH source_article AS (
    SELECT id FROM articles WHERE public_id = :public_id
),
article_classifications AS (
    SELECT article_id,
           MAX(confidence_score) AS max_confidence,
           jsonb_agg(jsonb_build_object(
               'classifier_type', classifier_type,
               'confidence_score', confidence_score,
               'reasoning', reasoning
           )) AS classifications
    FROM classifications
    GROUP BY article_id
),
related_scores AS (
    SELECT ae.article_id, COUNT(*) AS shared_entity_count
    FROM article_entities ae
    INNER JOIN article_entities src_ae ON src_ae.entity_id = ae.entity_id
    INNER JOIN source_article sa ON src_ae.article_id = sa.id
    WHERE ae.article_id != sa.id
    GROUP BY ae.article_id
)
SELECT
    a.public_id, a.url, a.title, a.section, a.published_date,
    ns.id AS news_source_id,
    NULL::text AS snippet,
    a.full_text,
    COALESCE(array_agg(DISTINCT e.name) FILTER (WHERE e.id IS NOT NULL), '{}') AS entities,
    COALESCE(ac.classifications, '[]'::jsonb) AS classifications
FROM related_scores rs
JOIN articles a ON a.id = rs.article_id
JOIN news_sources ns ON a.news_source_id = ns.id
LEFT JOIN article_classifications ac ON a.id = ac.article_id
LEFT JOIN article_entities ae ON a.id = ae.article_id
LEFT JOIN entities e ON ae.entity_id = e.id
GROUP BY a.id, a.public_id, a.url, a.title, a.section, a.published_date,
         a.full_text, ns.id, ac.classifications, ac.max_confidence, rs.shared_entity_count
ORDER BY rs.shared_entity_count DESC, COALESCE(ac.max_confidence, 0) DESC, a.published_date DESC
LIMIT :limit;
