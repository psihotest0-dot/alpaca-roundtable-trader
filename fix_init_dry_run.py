with open("src/trading_system/execution/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "        self.settings = settings\n        self._lock = threading.Lock()",
    "        self.settings = settings\n        self.is_dry_run = is_dry_run\n        self._lock = threading.Lock()"
)

with open("src/trading_system/execution/engine.py", "w") as f:
    f.write(content)
