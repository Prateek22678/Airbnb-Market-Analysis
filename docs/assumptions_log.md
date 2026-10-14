# Assumptions Log

This document records all methodological decisions, data cleaning rules, statistical assumptions, and business logic applied throughout the Airbnb Market Analysis project for Austin, TX.

---

| ID | Date | Category | Decision / Assumption | Rationale | Impact / Limitations |
|---|---|---|---|---|---|
| ASM-001 | 2026-10-14 | Setup | Target market: Austin, Texas (Inside Airbnb snapshot 2026-09-17) | High-volume market with vibrant tourism (SXSW, ACL, F1), predominantly English reviews (>99%), rich geospatial and seasonal data. | Analysis reflects post-summer 2026 market dynamics. |
| ASM-002 | 2026-10-14 | Architecture | SQLite relational model with Python processing pipeline | SQLite is lightweight, serverless, zero-configuration, fully ACID compliant, and easily queryable via standard SQL tools. | Query execution runs in-memory or on local disk; adequate for single-market scale. |
