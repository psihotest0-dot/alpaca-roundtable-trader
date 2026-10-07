with open("src/trading_system/execution/engine.py", "r") as f:
    content = f.read()

# Add parameter
content = content.replace("    def __init__(\n        self,\n        broker: AlpacaBroker,\n        order_repo: OrderRepository,\n        state_repo: StateRepository,\n        settings: Settings,\n    ):", "    def __init__(\n        self,\n        broker: AlpacaBroker,\n        order_repo: OrderRepository,\n        state_repo: StateRepository,\n        settings: Settings,\n        is_dry_run: bool = False,\n    ):")

content = content.replace(
    "        self.settings = settings\n\n",
    "        self.settings = settings\n        self.is_dry_run = is_dry_run\n\n"
)

# Fix submit_order (around line 98: `try:` followed by `# Build alpaca request`)
replace_submit = """            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would submit order {client_order_id} for {intent.symbol}")
                broker_order = BrokerOrder(
                    broker_id=f"dry_run_{client_order_id}",
                    client_order_id=client_order_id,
                    symbol=str(intent.symbol),
                    side=intent.side,
                    type=intent.type,
                    status=OrderStatus.SUBMITTED,
                    quantity=intent.quantity,
                    filled_quantity=Decimal("0"),
                    limit_price=intent.limit_price,
                    stop_price=intent.stop_price,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                )
                try:
                    self.order_repo.save_broker_order(broker_order)
                except Exception as e:
                    logger.error(f"Failed to persist dry-run broker order {client_order_id}: {e}")
                    return None
                return broker_order

            try:
                # Build alpaca request"""

content = content.replace("            try:\n                # Build alpaca request", replace_submit, 1)

# Fix cancel_bracket_order (around line 346: `try:` followed by `alpaca_order = self.broker.get_order_by_client_id(client_order_id)`)
replace_cancel = """            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would cancel order {client_order_id}")
                return

            try:
                alpaca_order = self.broker.get_order_by_client_id(client_order_id)"""

content = content.replace("            try:\n                alpaca_order = self.broker.get_order_by_client_id(client_order_id)", replace_cancel, 1)


# Find the emergency action
replace_emergency = """                    if self.is_dry_run:
                        logger.critical(f"[DRY-RUN] Would submit emergency market close for {sym}")
                        continue

                    # Documented action: In paper trading, submit a market sell order to close it out immediately
"""
content = content.replace("""                    # Documented action: In paper trading, submit a market sell order to close it out immediately\n""", replace_emergency)

with open("src/trading_system/execution/engine.py", "w") as f:
    f.write(content)
