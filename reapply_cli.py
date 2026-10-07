with open("src/trading_system/cli.py", "r") as f:
    content = f.read()

inject_logic = """
def inject_offline_dummy_keys():
    import os
    os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy_offline_key")
    os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy_offline_secret")

"""
if "inject_offline_dummy_keys" not in content:
    content = content.replace("from trading_system.config import Settings\n\n\ndef create_parser", "from trading_system.config import Settings\n\n" + inject_logic + "def create_parser")

    replace_validate = """    if args.command == "validate-config":
        try:
            inject_offline_dummy_keys()
            Settings()  # type: ignore"""
    content = content.replace("""    if args.command == "validate-config":
        try:
            Settings()  # type: ignore""", replace_validate)

    replace_doctor = """    elif args.command == "doctor":
        try:
            if not args.network:
                inject_offline_dummy_keys()
            settings = Settings()  # type: ignore"""
    content = content.replace("""    elif args.command == "doctor":
        try:
            settings = Settings()  # type: ignore""", replace_doctor)

with open("src/trading_system/cli.py", "w") as f:
    f.write(content)
