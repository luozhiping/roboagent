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
    print(fileName, data[0])
    # if '__pycache__' in fileName:
    #   mycursor.execute("delete from repos where id=%s" % data[0])
  mydb.commit()
download()