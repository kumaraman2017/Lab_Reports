import app
import json

print("---- TEST NO DATES ----")
res1 = app.get_test_history("A1C")
print(json.dumps(res1, indent=2))

print("\n---- TEST DATES ----")
res2 = app.get_test_history("A1C", "2024-01-01", "2024-03-31")
print(json.dumps(res2, indent=2))

