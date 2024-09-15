import pymysql

def connect_to_db():
    """Establish a connection to the database"""
    return pymysql.connect(
        host = '',
        port = 3306,
        user = '',
        password = '',
        database = ''
    )

def close_connection(connection):
    """Close the connection to the database"""
    connection.close()

if __name__ == '__main__':
    connection = connect_to_db()
    print(connection)
    close_connection(connection)
