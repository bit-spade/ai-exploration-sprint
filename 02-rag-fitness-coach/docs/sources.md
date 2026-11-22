# RAG Fitness Coach — Source Documents

All documents are freely available and either public domain, CC-licensed, or
from official government/health organisations.

---

## Pillar 1: Exercise Library — Muscle Groups, Purpose & Form

| # | Title | Source | URL | License | Format | Status |
|---|-------|--------|-----|---------|--------|--------|
| 1 | WHO Guidelines on Physical Activity | World Health Organization | https://www.who.int/news-room/fact-sheets/detail/physical-activity | CC BY-NC-SA 3.0 IGO | Web | active |
| 2 | Strength and Resistance Training — Principles & Physiology | Wikipedia (Open Knowledge) | https://en.wikipedia.org/wiki/Strength_training | CC BY-SA 4.0 | Web | active |
| 3 | Exercise Physiology — Muscular & Cardiovascular Adaptations | Wikipedia (Open Knowledge) | https://en.wikipedia.org/wiki/Exercise_physiology | CC BY-SA 4.0 | Web | active |

---

## Pillar 2: Health-Issue Exercises & Rehabilitation

| # | Title | Source | URL | License | Format | Status |
|---|-------|--------|-----|---------|--------|--------|
| 4 | Low Back Pain — Exercise, Prevention and Rehabilitation | Wikipedia (Medical Editors) | https://en.wikipedia.org/wiki/Low_back_pain | CC BY-SA 4.0 | Web | active |
| 5 | Aerobic Exercise — Cardiovascular Health & Metabolic Benefits | Wikipedia (Medical Editors) | https://en.wikipedia.org/wiki/Aerobic_exercise | CC BY-SA 4.0 | Web | active |

---

## Pillar 3: Nutrition

| # | Title | Source | URL | License | Format | Status |
|---|-------|--------|-----|---------|--------|--------|
| 6 | WHO — Healthy Diet & Micronutrient Guidance | World Health Organization | https://www.who.int/news-room/fact-sheets/detail/healthy-diet | CC BY-NC-SA 3.0 IGO | Web | active |
| 7 | Sports Nutrition — Macronutrients, Protein, Hydration & Timing | Wikipedia (Open Knowledge) | https://en.wikipedia.org/wiki/Sports_nutrition | CC BY-SA 4.0 | Web | active |

---

## Notes

- PDFs are downloaded to `data/raw/` (gitignored).
- Web pages are fetched at ingestion time via `WebBaseLoader`.
- Update `Status` from `planned` → `ingested` once added to the vector store.