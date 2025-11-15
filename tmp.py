from datasets import load_dataset

sst2 = load_dataset("glue", "sst2")

print("train:", len(sst2["train"]))
print("validation:", len(sst2["validation"]))
print("test:", len(sst2["test"]))