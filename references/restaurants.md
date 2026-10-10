# Restaurants (Fodor's / Zagat)

| | |
|---|---|
| Domain | Restaurant listings from the Fodor's and Zagat guides |
| Source | Kaggle [`adrianevensen/restaurants-entity-resolution`](https://www.kaggle.com/datasets/adrianevensen/restaurants-entity-resolution), file `restaurants.csv`; originally the RIDDLE "Restaurant" set by Sheila Tejada |
| Licence | Listed as "Unknown" on Kaggle |
| Records → entities | 864 → 752 |
| Entity sizes | 640 singletons; 112 duplicate pairs; largest entity 2 records |
| Ground truth | The `cluster` column |
| Fields kept | name, address, city, phone, category |

Notes:

- A small, easy set: used as a smoke test, not to rank methods.
- The raw file lists each entity's records next to each other, so the prepared records are shuffled.
