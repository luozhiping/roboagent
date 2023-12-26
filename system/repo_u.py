import mysql.connector

import os
import hashlib
import base64
from database import Database

mydb = mysql.connector.connect(
  host="rm-bp1o2l4qa4w1722mh6o.mysql.rds.aliyuncs.com",
  database="ai",
  user="ai",
  password="Ai123456"
)

database = Database()

print(mydb)
mycursor = mydb.cursor()

skip_directory = ['.idea', 'data', 'wandb', '.git']

allow_suffix = ['.py','.sh']

allow_file = []#['dataset_stats.pkl']#, 'policy_best.ckpt']

def upload():

  update_list = []
  insert_list = []
  delete_list = []

  # 1. 先更新一遍
  for root, dirs, files in os.walk('../', topdown=False):
    for name in files:
      print('----------------')
      skip = False
      for dir in skip_directory:
        # print(dir, root, name, (dir in root))
        if dir in root:
          skip = True
          break
      filename = os.path.join(root, name)
      if skip:
        print('skip1: %s' % filename)
        continue
      skip = True
      for suffix in allow_suffix:
        if filename.endswith(suffix):
          skip = False
          break
      for allowf in allow_file:
          print(allowf, filename)
          if allowf in filename:
              skip = False
              break
      if skip:
        print('skip2: %s' % filename)
        continue
      # if not name.endswith('.py'):
      #   continue
      content = open(filename, 'rb').read()
      md = hashlib.md5()  # 获取一个md5加密算法对象
      md.update(content)  # 制定需要加密的字符串
      md5 = md.hexdigest()
      print('文件:', filename, root, md5)
      bFilename = base64.b64encode(filename.encode('utf-8')).decode('utf-8')
      bContent = content
      datas = database.get_file_by_filename(bFilename)
      update = True
      # print(datas)
      if datas:
        data = datas[0]
        if data[3] == md5:
          update = False
        else:
          print('更新文件: %s, md5 %s, old md5 %s' % (filename, md5, data[3]))
          update_list.append(filename)
      else:
        print('插入新文件: %s, md5 %s' % (filename, md5))
        insert_list.append(filename)
      # mydb.commit()
      if update:
        print('需要更新: %s' % filename)
        # mycursor.close()
        mycursor.execute("REPLACE INTO repos (name, content, md5) values (%s, %s, %s)", (bFilename, bContent, md5))
        mydb.commit()
      else:
        pass
        # print('不需要更新: %s' % filename)

  # 2. 删除不要的
  mycursor.execute("SELECT * FROM repos;")
  datas = mycursor.fetchall()
  for data in datas:
    fileName = base64.b64decode(data[1]).decode('utf-8')
    if not os.path.exists(fileName):
      print('%s 已删除，同步' % fileName)
      mycursor.execute("DELETE from repos where name = %s", (data[1], ))
      # print(data, fileName)
      delete_list.append(fileName)
      mydb.commit()
  print('insert: %s' % insert_list)
  print('update: %s' % update_list)
  print('delete: %s' % delete_list)

upload()
