import numpy as np
text=[]
with open("test.txt", "r", encoding="UTF-8") as f:
    text=[i.strip() for i in f if i.strip()]
final=[]
with open ("res.txt", "w") as f:
    for i in range(len(text)):
        if i <31:
            f.write("data_timit["+str(i)+ "] = new Pair(\""+text[i].split("(")[1].replace(")", "")+"\",\""+text[i].split("(")[0].strip()+"\");\n")
        elif 31<=i<42:
            f.write("data_google["+str(i-31)+ "] = new Pair(\""+text[i].split("(")[1].replace(")", "")+"\",\""+text[i].split("(")[0].strip()+"\");\n")
        else:
            f.write("data_siri["+str(i-42)+ "] = new Pair(\""+text[i].split("(")[1].replace(")", "")+"\",\""+text[i].split("(")[0].strip()+"\");\n")
