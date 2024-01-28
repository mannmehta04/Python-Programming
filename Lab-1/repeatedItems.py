t1 = (10,20,40,20,40)
r = []

for i in t1:
	if t1.count(i) > 1 and i not in r:
		r.append(i)

if r:
	print(r)
else:
	print("nothing duplicated")