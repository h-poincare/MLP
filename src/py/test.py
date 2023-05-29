



# l = [1,3,12,4,5]
# print(l)
# print(reversed(l))

# import numpy as np 
# a=np.array([[1,2,3],[0,0,0]])
# b=np.array([[1,5,1],[0,0,0]])
# c=np.multiply(a,b)
# print(c)
# print(np.sum(c))


# import numpy as np

# pool_arr = np.zeros((2,2))
# conv = np.array([[1,2,3,6,2,1], [0,4,2,2,1,0], [0,3,2,6,3,2],
#                  [1,4,7,3,6,2], [0,4,5,3,8,3], [0,2,4,7,3,2]])

# print(conv)
# print(conv[0:2, 0:2])
# print(np.max(conv[0:2, 0:2]))
# print(conv[0:2, 2:4])
# print(conv[0:2, 4:6])
# print(conv[2:4, 0:2])

import numpy as np
def relu(x):
    return max(0, np.max(x)) 

# arr = np.array([[-1, 2, 3],[4, -5, 6]])
# # relu_func = lambda x: relu(x) 
# # vfunc = np.vectorize(relu_func)
# # print(vfunc(arr))
# print(arr)
# print(np.pad(arr, 1, mode='constant', constant_values=(0)))

# a = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
# print(a)
# print(a.shape)


# print(np.pad(a, 1, mode='constant', constant_values=(0)))



# class Car:
#     # a="fname"
#     # b="lname"
#     def __init__(self):
#         self.a="fname"
#         self.b="lname"

# c1 = Car()
# c2 = Car()

# print(c1.a)
# print(c2.a)

# c1.a = "NEW"
# print(c1.a)
# print(c2.a)

# Car.a = "NEW"
# print(c1.a)
# print(c2.a)

# Car("f1")


# class Foo:
#     static_var = 'every instance has access'

#     def __init__(self,name):
#         self.instance_var = 'I am %s' % name

#     def printAll(self):
#         print('self.instance_var = %s' % self.instance_var) 
#         print('self.static_var = %s' % self.static_var) 
#         print('Foo.static_var = %s' % Foo.static_var) 

# f1 = Foo('f1')

# f1.printAll()

# f1.static_var = 'Shadowing static_var'

# f1.printAll()

# f2 = Foo('f2')

# f2.printAll()

# Foo.static_var = 'modified class'

# f1.printAll()
# f2.printAll()

# class OracleDB():

#     def __init__(self, host, user, pw, port):
#         self.host = host
#         self.user = user
#         self.pw = pw
#         self.port = port

#     def connection(self):
#         pass
#         #create connection to OracleDB

#     def query(self, sql, dest):
#         pass
#         #pass SQL query to conn.. save query output to csv

#     def close(self):
#         pass 
#         #close Oracle DB connection 

    
        

