import mysql.connector

class Database:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="rm-bp1o2l4qa4w1722mh6o.mysql.rds.aliyuncs.com",
            database="ai",
            user="ai",
            password="Ai123456"
        )

        self.mycursor = self.mydb.cursor()

    def get_file_by_filename(self, filename):
        self.mycursor.execute("SELECT * FROM repos where name = %s;", (filename,))
        datas = self.mycursor.fetchall()
        return datas