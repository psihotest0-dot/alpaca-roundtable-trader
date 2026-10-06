# Operations

## Local Development
- Install dependencies using `poetry install`.
- Run tests using `poetry run pytest`.
- Execute the application entry point via `poetry run python -m trading_system`.

## Docker
- The application container encapsulates only the trading logic.
- Use `docker compose run --rm app <command>` for isolated execution.

## Future Functionality vs Available functionality
Currently, the codebase provides the foundational skeleton. It supports importing, CLI help, and structural components.
Future updates will include market data ingestion, strategy execution, order routing, and the AnythingLLM connection.

*Note: The host Chroma server must be managed separately from this application's Docker compose setup.*
