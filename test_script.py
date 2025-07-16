# test_script.py
import datetime

timestamp = datetime.datetime.utcnow().isoformat()
with open("output.txt", "w") as f:
    f.write(f"✅ This file was auto-generated on the Azure VM at {timestamp} UTC\n")

print("Script executed, output.txt created.")
