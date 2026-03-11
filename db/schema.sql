CREATE TABLE IF NOT EXISTS moex_bonds_securities (
    secid TEXT PRIMARY KEY,
    shortname TEXT,
    name TEXT,
    isin TEXT,
    emitent_id TEXT,
    emitent_title TEXT,
    emitent_inn TEXT,
    issue_date DATE,
    maturity_date DATE,
    coupon_percent NUMERIC,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bond_emitters (
    emitter_key TEXT PRIMARY KEY,
    emitent_title TEXT,
    emitent_inn TEXT,
    bonds_count INTEGER NOT NULL,
    first_issue_date DATE,
    last_maturity_date DATE,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE OR REPLACE VIEW bond_emitters_market_view AS
SELECT
    emitent_title,
    emitent_inn,
    bonds_count,
    first_issue_date,
    last_maturity_date,
    updated_at
FROM bond_emitters;
