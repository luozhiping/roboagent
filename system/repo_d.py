import mysql.connector
import os
import hashlib
import base64

mydb = mysql.connector.connect(
  host="rm-bp1o2l4qa4w1722mh6o.mysql.rds.aliyuncs.com",
  database="ai",
  user="ai",
  password="Ai123456"
)

print(mydb)
mycursor = mydb.cursor()

skip_directory = ['.idea']

def download():
  mycursor.execute("SELECT * FROM repos;")
  datas = mycursor.fetchall()

  update_list = []
  insert_list = []
  delete_list = []

  for data in datas:
    print('------------')
    fileName = base64.b64decode(data[1]).decode('utf-8')
    folder = os.path.dirname(fileName)
    print(fileName, folder)
    if not os.path.exists(fileName):
      os.makedirs(folder, exist_ok=True)
      content = data[2]
      open(fileName, 'wb').write(content)
      print('插入文件: %s' % fileName)
      insert_list.append((fileName))
    else:
      content = open(fileName, 'rb').read()
      md = hashlib.md5()  # 获取一个md5加密算法对象
      md.update(content)  # 制定需要加密的字符串
      md5 = md.hexdigest()
      if md5 == data[3]:
        continue
      else:
        content = data[2]
        open(fileName, 'wb').write(content)
        print('更新文件: %s' % fileName)
        update_list.append(fileName)
      # open(fileName, 'wb').write(data[2])
      # print('%s 已删除，同步' % fileName)
      # mycursor.execute("DELETE from repos where name = %s", (data[1],))
      # # print(data, fileName)
      # mydb.commit()


  for root, dirs, files in os.walk('../repo/', topdown=False):
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
        print('skip: %s' % filename)
        continue
      bFilename = base64.b64encode(filename.encode('utf-8')).decode('utf-8')
      mycursor.execute("SELECT * FROM repos where name = %s;", (bFilename,))
      datas = mycursor.fetchall()
      if not datas:
        print('删除文件: %s' % filename)
        os.remove(filename)
        delete_list.append(filename)
  print('insert: %s' % insert_list)
  print('update: %s' % update_list)
  print('delete: %s' % delete_list)



download()