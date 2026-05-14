# 모듈 생성
def add(a, b):
    return a + b

def sub(a, b):
    return a - b

# 모듈 안 기능들이 잘 만들어졌는지 확인
# mod1이 실행 될 때만 if문 실행
if __name__ == "__main__":
    print(add(3,4))
    print(sub(3,4))