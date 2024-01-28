def prod(t1):
	
	p=1

	for i in t1:
		p *= i
	
	return p

t1 = (2,7,4)
print(prod(t1))