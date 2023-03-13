import time
from glob import glob
import shutil,os

#去除名字的特殊符号，统一序号视频文件命名

result = list(glob("E:/Projects/wav2lip_train/data/original_data/*",recursive=False))
file_num = 0
result_list = []
base_path = "E:/Projects/wav2lip_train/data/original_data/"

for each in result:
    file_num +=1
    new_position ="{0}{1}".format( int(time.time()),file_num)
    result_list.append(new_position)
    shutil.move(each, os.path.join(base_path,new_position+".mp4"))
    pass

file_list_path = os.path.join("E:/Projects/wav2lip_train/filelists","train.txt")
with open(file_list_path,'w',encoding='utf-8') as fi:
    fi.write("\n".join(result_list))
    fi.flush()
