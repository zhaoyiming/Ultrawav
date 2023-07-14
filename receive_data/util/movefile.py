import os, shutil
import re
basedir="D:/record_data/original/"
newdir="D:/record_data/wav/"
alldir = os.listdir(basedir)
allfilename=[]

for dir in alldir:
    root = basedir+dir+'/'
    print(root)

    # collect all filename
#     for path, subdirs, files in os.walk(root):
#         for name in files:
#             if name.endswith('.wav'):
#                 allfilename.append(os.path.join(path, name))
#
# print(allfilename)
# with open("C:/Users/ming/PycharmProjects/receive/test.txt", "w") as f:
#     for i in allfilename:
#         f.write(i+'\n')
#     f.close()
#



    # move dir
    # for path, subdirs, files in os.walk(root):
    #     for name in files:
    #         if name.endswith('.wav'):
    #             folder = root + name.split("-")[0]
    #             if not os.path.exists(folder):
    #                 os.mkdir(folder)
    #             if not os.path.exists(os.path.join(folder, name)):
    #                 shutil.copyfile(os.path.join(path, name), os.path.join(folder, name))

    # move to new dir
    new_root=newdir+dir+'/'
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    for path, subdirs, files in os.walk(root):
        for name in files:
            print(name)
            if name.endswith('.pcm'):
                folder = new_root + name.split("-")[0]
                if not os.path.exists(folder):
                    os.mkdir(folder)
                if not os.path.exists(os.path.join(folder, name)):
                    shutil.copyfile(os.path.join(path, name), os.path.join(folder, name))




    # # del raw dir
    # del_dir = os.listdir(root)
    # for i in del_dir:
    #     if bool(re.search(r"[a-zA-Z]", i)):
    #         if i.endswith('.wav'):
    #             os.remove(root + i)
    #         else:
    #             shutil.rmtree(root + i)
