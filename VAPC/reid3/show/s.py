import os, shutil
for i in [15,25,85,920,890,815,665,640,525,470,435,370,355]:
    fp=open('show'+str(i)+'.txt')
    content=fp.readlines()
    if not os.path.exists(str(i)):
        os.makedirs(str(i))
    for c in content:
        print(c)
        shutil.copy(c.replace('\n', '').replace('\r', ''), os.path.join(str(i), c.split('/')[5]))