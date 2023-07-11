import mysql.connector

# Connect to server
cnx = mysql.connector.connect(
    host="Erik",
    port=3306,
    user="root",
    password="304501xie")

# Get a cursor
cur = cnx.cursor()

# Execute a query
cur.execute("SELECT CURDATE()")

# Fetch one result
row = cur.fetchone()
print("Current date is: {0}".format(row[0]))

# Close connection
cnx.close()