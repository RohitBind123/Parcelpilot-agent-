-- ParcelPilot structured store (Tier 0: the workbook).
--
-- Two conventions carry weight here.
--
-- Timestamps are TEXT in ISO 8601 *with an offset*. SQLite has no datetime
-- type, and a naive string would resolve differently on every host, silently
-- shifting every cancellation window and SLA clock.
--
-- Nullable columns stay nullable. A missing pickup time means the pickup is
-- not recorded, which is a different fact from "picked up at the epoch", and a
-- missing contract file means no agreement, not a contract named "". Coercing
-- either to a default is the misleading-zero failure one layer below the UI.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS clause_topics;
DROP TABLE IF EXISTS clauses;
DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS meta;

-- Build provenance. `as_of` is read from the workbook README sheet and is the
-- only "now" the system has; a test asserts it matches the configured value so
-- config and data cannot drift apart.
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE accounts (
    account_id      TEXT PRIMARY KEY,
    account_name    TEXT    NOT NULL,
    plan            TEXT    NOT NULL CHECK (plan IN ('Enterprise', 'Growth', 'Standard')),
    status          TEXT    NOT NULL,
    csm             TEXT,
    -- NULL means no signed agreement in the pack, which is a governing fact:
    -- the account falls back to general policy with no override available.
    contract_file   TEXT,
    premium_support INTEGER NOT NULL CHECK (premium_support IN (0, 1)),
    notes           TEXT
) WITHOUT ROWID;

CREATE TABLE orders (
    order_id                  TEXT PRIMARY KEY,
    account_id                TEXT    NOT NULL REFERENCES accounts (account_id),
    carrier                   TEXT    NOT NULL,
    status                    TEXT    NOT NULL
        CHECK (status IN ('DRAFT', 'BOOKED', 'PICKED_UP', 'DELIVERED', 'CANCELLED')),
    booked_at                 TEXT    NOT NULL,
    pickup_window_start       TEXT,
    pickup_window_end         TEXT,
    -- NULL means ParcelPilot has no pickup confirmation. Given KI-211, that is
    -- not the same as "the parcel was not collected".
    pickup_actual_at          TEXT,
    shipment_fee_inr          REAL,
    carrier_fault             INTEGER NOT NULL DEFAULT 0 CHECK (carrier_fault  IN (0, 1)),
    customer_fault            INTEGER NOT NULL DEFAULT 0 CHECK (customer_fault IN (0, 1)),
    cancellation_requested_at TEXT,
    notes                     TEXT
) WITHOUT ROWID;

CREATE TABLE tickets (
    ticket_id                TEXT PRIMARY KEY,
    account_id               TEXT NOT NULL REFERENCES accounts (account_id),
    created_at               TEXT NOT NULL,
    status                   TEXT NOT NULL,
    subject                  TEXT NOT NULL,
    description              TEXT,
    channel                  TEXT,
    -- Drives my_queue. The two values present are 'Maya' and 'Rohit'.
    assigned_to              TEXT,
    last_customer_message_at TEXT,
    -- Tier 5. Present on closed tickets only, and both of the two in the pack
    -- are wrong. Retained deliberately so the contradiction can be detected.
    historical_resolution    TEXT
    -- Deliberately absent: severity and first_response_at. Neither exists in
    -- the workbook. Severity is derived from Policy v3 s2 (D23), and a real
    -- first-response breach is therefore not measurable - only elapsed time
    -- against the target is.
) WITHOUT ROWID;

CREATE INDEX idx_orders_account       ON orders  (account_id);
CREATE INDEX idx_orders_status        ON orders  (status);
CREATE INDEX idx_tickets_account      ON tickets (account_id);
CREATE INDEX idx_tickets_assigned_to  ON tickets (assigned_to);
CREATE INDEX idx_tickets_status       ON tickets (status);
-- The ops scan reads open tickets per account; the composite serves it directly.
CREATE INDEX idx_tickets_status_account ON tickets (status, account_id);


-- ---------------------------------------------------------------------------
-- The clause registry: the authority spine (Tiers 1-5).
--
-- Precedence is a sort on `tier` over rows this table already carries, rather
-- than a judgement made over whatever retrieval happened to return. That is
-- the difference between an override that is provable and one that is
-- contingent on recall.
--
-- `account_id` does double duty. NULL means the clause applies to every
-- account; a value means the clause is both private to that account and its
-- Tier 1 authority. One predicate therefore satisfies the ACL and the
-- precedence rule at once.

CREATE TABLE clauses (
    clause_id      TEXT PRIMARY KEY,
    doc_id         TEXT    NOT NULL,
    doc_title      TEXT    NOT NULL,
    clause_ref     TEXT    NOT NULL,
    title          TEXT    NOT NULL,
    tier           INTEGER NOT NULL CHECK (tier BETWEEN 0 AND 5),
    -- NULL = applies to all accounts. Otherwise this clause is private to it.
    account_id     TEXT             REFERENCES accounts (account_id),
    status         TEXT    NOT NULL,
    effective_from TEXT,
    effective_to   TEXT,
    superseded_by  TEXT,
    -- Typed values the calculators read. JSON because the shape differs per
    -- topic; the reviewed baseline in clause_params_baseline.yaml is what
    -- keeps it honest.
    params         TEXT    NOT NULL DEFAULT '{}',
    text           TEXT    NOT NULL
) WITHOUT ROWID;

-- Topics are a join table rather than a JSON array on `clauses`, because the
-- resolver's central query is "every clause about this subject, visible to
-- this account, ordered by tier" - and that wants an index.
CREATE TABLE clause_topics (
    clause_id TEXT NOT NULL REFERENCES clauses (clause_id),
    topic     TEXT NOT NULL,
    PRIMARY KEY (clause_id, topic)
) WITHOUT ROWID;

CREATE INDEX idx_clauses_tier         ON clauses (tier);
CREATE INDEX idx_clauses_account      ON clauses (account_id);
CREATE INDEX idx_clauses_doc          ON clauses (doc_id);
CREATE INDEX idx_clause_topics_topic  ON clause_topics (topic);
