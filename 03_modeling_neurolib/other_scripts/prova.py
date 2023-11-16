import sys 

random = sys.argv[1].lower()

# Mapping dictionary
mapping = {
    "true": True,
    "false": False
}

condition = mapping.get(random)

if not condition:
    print("This is not random!")
else:
    print("This is random!")