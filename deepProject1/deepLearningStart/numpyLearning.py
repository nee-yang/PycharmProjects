import numpy as np

a = np.array([1, 2, 3])
print(a.shape)
print(type(a))

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)
print(b)

print(np.ones((1, 2)))
print(np.full((2, 2), 7))
print(np.eye(2))
print(np.random.random((2, 2)))

print('--------------------切片(Slicing)-------------------------')
a = np.arange(12).reshape((3, 4))
print(a)
print(a.T)
print(a[:2, 1:3])

