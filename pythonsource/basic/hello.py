print("hello")

# 줄단위 실행 => 특정 행에서 실행 오류 발생 시 프로그램 정지됨
# 파이썬 자료형
# 정수형, 문자형, 불린형, 리스트형, 튜블, 딕셔너리, set
# 리스트, 튜블, 딕셔너리> 문자형> 정수형, 불린형

# 변수 : 프로그렘 내부에서 값을 담아놓기 위한 공간(이름 지정)
# 변수 a 지정
a = 123

print(a)

multiline = """
Life is too short
You need python
"""
print(multiline)

# import mod1

# print(mod1.add(9,3))
# print(mod1.sub(9,3))

# from mod1 import add

# print(add(4,5))

# *: 모두

from mod1 import *

print(add(4,5))
print(sub(4,5))
