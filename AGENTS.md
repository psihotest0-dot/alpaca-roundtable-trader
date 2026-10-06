# Agent Rules and Guidelines

## Architecture & Safety Rules
- **Paper Trading Only**: The MVP focuses strictly on paper trading. Live execution is out of scope.
- **Order Execution**: The Strategy does NOT send broker orders. The Oracle does NOT send orders and does NOT size positions. Only the Execution Engine can send orders.
- **Risk Engine**: Has the final say on approving any entry.
- **Broker Source of Truth**: The broker is the single source of truth for real positions and orders.
- **Independence**: Defensive exits and reconciliation mechanisms operate entirely independent of the Oracle.
- **Secrets Management**: NEVER output API keys or tokens in logs, test fixtures, or Git.
- **Testing**: No real network requests in standard tests.
- **LLM Safety**: Do NOT execute arbitrary code or instructions from LLM responses.
- **Validation**: Verify all SDK methods and API endpoints against the official documentation for the installed versions; do not use hallucinated endpoints.

## Chroma and AnythingLLM Architecture
- **Data Flow**: Trading App -> AnythingLLM API -> Local Chroma Server.
- **Chroma Setup**: The Trading App does not connect to Chroma directly.
- **Host Process**: Chroma runs as a host process outside of Docker.
- **Storage**: Chroma data is permanently stored on the host machine, OUTSIDE of this Git repository.
  - Linux/macOS: `~/opper-data/chroma`
  - Windows: `C:\opper-data\chroma`
- **Docker Constraints**:
  - Do NOT add Chroma as a service, image, or container in `compose.yaml`.
  - Do NOT mount the Chroma directory into the application containers.
  - Bind mounts are not a substitute for the host process setup. Both Chroma and its data must remain entirely outside Docker.
- **Networking**:
  - AnythingLLM can run in Docker, connecting to the host Chroma via network.
  - Do NOT use `localhost` or `127.0.0.1` inside a container to refer to the host Chroma.
  - Chroma must NOT be publicly accessible.
- **Independence**: Docker Compose does NOT start the host Chroma. Missing AnythingLLM or Chroma must not prevent package imports, CLI `--help`, or tests from passing.
- **No Client Integration**: Do NOT add a Chroma client library dependency to the trading app.
