import os
v = 0
files = os.listdir('pictures')
for f in files:
    os.rename('pictures/' + str(f), 'pictures/test_' + str(v) + '.png')
    v +=1