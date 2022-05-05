# Check if we can round down the row and col index to isolate
# the box it is in
# 0  |  1  |  2  |
# 3  |  4  |  5  |
# 6  |  7  |  8  |

if not None:
    print("Hi")

for i in range(0,10):
    val = int(i/3)
    print(str(i) + " | " + str(val))

# Row & Col start from 0
row = 3 
col = 4

box = (int(row/3)*3) + int(col/3)
print ("Box:" + str(box))
print("")
# Convert box to row, col
for box in range(9):

    row = int(box / 3)*3
    col = box % 3 * 3

    print ("Box: " + str(box) + "  |" + str(row) + " | " + str(col))