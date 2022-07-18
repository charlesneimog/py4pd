from inspect import signature
sig = signature(function1)
str(sig) 
params = sig.parameters 
print(len(params))