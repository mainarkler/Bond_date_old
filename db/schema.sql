CREATE TABLE IF NOT EXISTS bonds (
    secid TEXT PRIMARY KEY,
    shortname TEXT,
    name TEXT,
    isin TEXT,
    emitent_id TEXT,
    emitent_title TEXT,
    emitent_inn TEXT,
    coupon_percent NUMERIC,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
