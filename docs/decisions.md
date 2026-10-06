# Decisions

- **Dependency Manager**: We chose `poetry` to ensure reproducible builds across local, CI, and Docker environments via `poetry.lock`.
- **Database**: SQLite is used for the MVP to minimize setup friction and keep local persistence simple.
- **Scope**: The MVP strictly implements paper trading. Live trading features have been explicitly excluded.
- **Chroma**: Chroma must run on the host machine to maintain persistent data (`~/opper-data/chroma`) outside the Git repo and Docker lifecycle.
