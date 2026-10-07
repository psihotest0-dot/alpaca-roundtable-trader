with open("src/trading_system/cli.py", "r") as f:
    content = f.read()

content = content.replace("def inject_offline_dummy_keys():", "def inject_offline_dummy_keys() -> None:")

with open("src/trading_system/cli.py", "w") as f:
    f.write(content)
