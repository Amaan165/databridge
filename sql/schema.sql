-- ============================================================================
-- DataBridge — Star Schema for Cross-City Restaurant Inspection Warehouse
-- DuckDB DDL  |  Data Engineering Spring 2026
--
-- Tables:
--   dim_restaurants          Deduplicated restaurant/business dimension
--   dim_geography            City + sub-geography reference
--   fact_inspections         One row per inspection (aggregated from violations)
--   dim_violations           One row per violation instance
--   dim_violation_taxonomy   Category labels from clustering  (Week 2 — Rithujaa)
--   dim_violation_crosswalk  Cross-city violation matches     (Week 2 — Rithujaa)
--
-- Surrogate key strategy:
--   All surrogate PKs are auto-incrementing integers (BIGINT).
--   Source IDs (camis, license_no, inspection_id) are preserved for traceability.
-- ============================================================================

-- ── Geography dimension ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_geography (
    geo_id              INTEGER PRIMARY KEY,
    city                VARCHAR NOT NULL,       -- 'nyc', 'chicago', 'boston'
    sub_geography       VARCHAR,                -- borough (NYC), neighborhood (Boston), 'CHICAGO' for Chicago
    latitude_centroid   DOUBLE,
    longitude_centroid  DOUBLE
);

-- ── Restaurant dimension ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_restaurants (
    restaurant_id       INTEGER PRIMARY KEY,
    source_id           VARCHAR NOT NULL,       -- camis (NYC), license_no (Chicago/Boston)
    name                VARCHAR,
    address             VARCHAR,
    latitude            DOUBLE,
    longitude           DOUBLE,
    city                VARCHAR NOT NULL,
    sub_geography       VARCHAR,                -- borough / neighborhood / city
    zipcode             VARCHAR,
    cuisine_type        VARCHAR,                -- NYC only; NULL for others
    geo_id              INTEGER REFERENCES dim_geography(geo_id),

    UNIQUE (city, source_id)
);

-- ── Inspection fact table ───────────────────────────────────────────────────
-- One row per inspection event (aggregated from violation-level data).
CREATE TABLE IF NOT EXISTS fact_inspections (
    inspection_id           INTEGER PRIMARY KEY,
    restaurant_id           INTEGER NOT NULL REFERENCES dim_restaurants(restaurant_id),
    source_inspection_id    VARCHAR,            -- original ID: inspection_id (CHI), camis+date (NYC), license_no+date (BOS)
    inspection_date         DATE NOT NULL,
    inspection_type         VARCHAR,            -- raw inspection type label
    is_reinspection         BOOLEAN DEFAULT FALSE,
    is_standard_inspection  BOOLEAN DEFAULT TRUE,   -- NYC only; TRUE for Chicago/Boston
    outcome_tier            VARCHAR,            -- 'Pass', 'Conditional', 'Fail', or NULL
    outcome_source          VARCHAR,            -- 'grade', 'score', or NULL (NYC only)
    score                   DOUBLE,             -- NYC numeric score; NULL for others
    grade                   VARCHAR,            -- NYC letter grade; NULL for others
    result_code             VARCHAR,            -- Boston result_code, Chicago results; NULL for NYC
    violation_count         INTEGER DEFAULT 0,
    critical_violation_count INTEGER DEFAULT 0, -- NYC critical_flag='Critical' count
    city                    VARCHAR NOT NULL
);

-- ── Violation dimension ─────────────────────────────────────────────────────
-- One row per individual violation instance.
CREATE TABLE IF NOT EXISTS dim_violations (
    violation_id            INTEGER PRIMARY KEY,
    inspection_id           INTEGER NOT NULL REFERENCES fact_inspections(inspection_id),
    violation_code          VARCHAR,            -- violation_code (NYC/BOS), violation_number (CHI)
    violation_description   VARCHAR,            -- violation_description (NYC/BOS), violation_category (CHI)
    violation_comment       VARCHAR,            -- violation_comment (CHI), comments (BOS); NULL for NYC
    severity                VARCHAR,            -- critical_flag (NYC), violation_severity (BOS); NULL for CHI
    city                    VARCHAR NOT NULL,
    taxonomy_category_id    INTEGER             -- FK to dim_violation_taxonomy; NULL until taxonomy loaded
);

-- ── Violation taxonomy (populated by Rithujaa, Task 2.3/2.4) ───────────────
CREATE TABLE IF NOT EXISTS dim_violation_taxonomy (
    taxonomy_category_id    INTEGER PRIMARY KEY,
    category_name           VARCHAR NOT NULL,    -- e.g. 'Pest/Vermin', 'Temperature Control'
    category_description    VARCHAR
);

-- ── Violation crosswalk (populated by Rithujaa, Task 2.4) ───────────────────
CREATE TABLE IF NOT EXISTS dim_violation_crosswalk (
    crosswalk_id            INTEGER PRIMARY KEY,
    violation_desc_city_a   VARCHAR NOT NULL,
    city_a                  VARCHAR NOT NULL,
    violation_desc_city_b   VARCHAR NOT NULL,
    city_b                  VARCHAR NOT NULL,
    similarity_score        DOUBLE,
    match_validated         BOOLEAN,            -- LLM confirmed (yes/no)
    taxonomy_category_id    INTEGER REFERENCES dim_violation_taxonomy(taxonomy_category_id)
);

-- ── Indexes for common query patterns ───────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_insp_city         ON fact_inspections(city);
CREATE INDEX IF NOT EXISTS idx_insp_date         ON fact_inspections(inspection_date);
CREATE INDEX IF NOT EXISTS idx_insp_outcome      ON fact_inspections(outcome_tier);
CREATE INDEX IF NOT EXISTS idx_insp_restaurant   ON fact_inspections(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_insp_reinspection ON fact_inspections(is_reinspection);
CREATE INDEX IF NOT EXISTS idx_viol_inspection   ON dim_violations(inspection_id);
CREATE INDEX IF NOT EXISTS idx_viol_taxonomy     ON dim_violations(taxonomy_category_id);
CREATE INDEX IF NOT EXISTS idx_rest_city         ON dim_restaurants(city);
CREATE INDEX IF NOT EXISTS idx_rest_geo          ON dim_restaurants(geo_id);
