import pymysql

def init_mysql():
    conn = pymysql.connect(host='127.0.0.1', user="root", passwd="123456a", db="ojdb", port=3306, charset="utf8")
    return conn.cursor()

def create_table():
    cur = init_mysql()
    sql = """
    create table dkt_demo1(
    id int auto_increment primary key,
    u_id int,
    p_id int,
    correct int 
    )engine=innodb default charset=utf8; 
    """
    cur.execute(sql)
    cur.close()

#读文件
def read_file(dataset_path):
    seqs_by_student = {}      #学生序列
    num_skills = 0            #题库题目数量
    conn = pymysql.connect(host='127.0.0.1', user="root", passwd="123456a", db="ojdb", port=3306, charset="utf8")
    with open(dataset_path, 'r') as f:  #读文件
        cur = conn.cursor()
        for line in f:
            fields = line.strip().split()    #去掉空格
            # 学生ID  问题ID    正确性
            student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
            sql = 'insert into dkt_demo1 (u_id,p_id,correct) values(%s,%s,%s)'
            cur.execute(sql,[student, problem, is_correct])
            conn.commit()

        cur.close()
    print("数据读取完成")

def get_Data():
    cur = init_mysql()
    try:
        with cur as cursor:
            sql = 'select * from dkt_demo1'
            cur.execute(sql)
            result = list(cur.fetchall())
            for datalist in result:
                print(datalist[1])
                print(datalist[2])
            # print(result)
    finally:
        cur.close()
if __name__ == '__main__':
    # read_file("../data/data.txt")
   # create_table()
   get_Data()