with open("src/trading_system/execution/engine.py", "r") as f:
    lines = f.readlines()

if "is_dry_run: bool" not in "".join(lines):
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "def __init__(":
            new_lines.append(line)
            while lines[i+1].strip() != "):":
                i += 1
                new_lines.append(lines[i])
            new_lines.append("        is_dry_run: bool = False,\n")
            i += 1
            new_lines.append(lines[i]) # "):"
        elif line.strip() == "self.settings = settings":
            new_lines.append(line)
            new_lines.append("        self.is_dry_run = is_dry_run\n")
        elif line.strip() == "def submit_order(self, intent: OrderIntent) -> Optional[BrokerOrder]:":
            new_lines.append(line)
            while True:
                i += 1
                new_lines.append(lines[i])
                if lines[i].strip() == "with self._lock:":
                    new_lines.append("""            if self.is_dry_run:
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
                return broker_order\n\n            try:\n""")
                    # Skip the original `try:` which we are wrapping or replacing?
                    # Wait, no, we shouldn't replace it if we just append before it.
                    # Let's do it carefully.
                    # Wait, the original code is:
                    #        with self._lock:
                    #            try:
                    # So we appended our logic AFTER `with self._lock:`
                    # So we should just skip the next `try:` line? No, we need it. Let's not write `try:\n` twice.
                    # Actually, if we just append, the original `try:` is at lines[i+1]. We'll just let the loop continue.
                    break
        elif line.strip() == "def cancel_bracket_order(self, client_order_id: str) -> None:":
            new_lines.append(line)
            while True:
                i += 1
                new_lines.append(lines[i])
                if lines[i].strip() == "with self._lock:":
                    new_lines.append("""            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would cancel order {client_order_id}")
                return\n""")
                    break
        elif line.strip() == "# Documented action: In paper trading, submit a market sell order to close it out immediately":
            new_lines.append("""                    if self.is_dry_run:
                        logger.critical(f"[DRY-RUN] Would submit emergency market close for {sym}")
                        continue\n""")
            new_lines.append(line)
        else:
            new_lines.append(line)
        i += 1

    with open("src/trading_system/execution/engine.py", "w") as f:
        f.writelines(new_lines)
