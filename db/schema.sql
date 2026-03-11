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

-- Indexes for faster aggregation and filtering on large datasets.
CREATE INDEX IF NOT EXISTS idx_moex_bonds_emitent_title ON moex_bonds_securities (emitent_title);
CREATE INDEX IF NOT EXISTS idx_moex_bonds_emitent_inn ON moex_bonds_securities (emitent_inn);
CREATE INDEX IF NOT EXISTS idx_moex_bonds_issue_date ON moex_bonds_securities (issue_date);
CREATE INDEX IF NOT EXISTS idx_moex_bonds_maturity_date ON moex_bonds_securities (maturity_date);

CREATE TABLE IF NOT EXISTS bond_emitters (
    emitter_key TEXT PRIMARY KEY,
    emitent_title TEXT,
    emitent_inn TEXT,
    bonds_count INTEGER NOT NULL,
    first_issue_date DATE,
    last_maturity_date DATE,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Indexes used by Streamlit sorting/filtering queries.
CREATE INDEX IF NOT EXISTS idx_bond_emitters_title ON bond_emitters (emitent_title);
CREATE INDEX IF NOT EXISTS idx_bond_emitters_bonds_count ON bond_emitters (bonds_count DESC);
CREATE INDEX IF NOT EXISTS idx_bond_emitters_last_maturity ON bond_emitters (last_maturity_date DESC);

CREATE OR REPLACE VIEW bond_emitters_market_view AS
SELECT
    emitent_title,
    emitent_inn,
    bonds_count,
    first_issue_date,
    last_maturity_date,
    updated_at
FROM bond_emitters;
