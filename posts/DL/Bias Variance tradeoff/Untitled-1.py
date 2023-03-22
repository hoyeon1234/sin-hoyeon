"""
변수
x=100 실행하면 메모리에 100이 저장됨.변수는 거기에 붙는(tagging) 별칭(꼬리표)느낌
"""

"""
대입연산자(=)
오른쪽에 있는 값을 왼쪽에 대입해라
"""
#ex)올바른예시
#100을 x에 대입해라
x = 100
"""

ex)잘못된예시
100 = x
100에 x를 대입해라=>말이안됨.100에 x를 대입 할 수는 없음
"""
"""
비교연산자(==),좌 우변이 같냐?
ex)
x=100
x == 100?
>>> True
ex)
100 == x?
>>> True

생성된 변수에는 얼마든지 다른 값을 저장할 수 있다.
x = 100
x = 200
print(x)
>>> 200

x=100
y=200
print(x,y)
>>> (100,200)

x=100
y=200
sum = x+y
print(sum)
>>>300
"""

#덧셈연산자
x=7
y=6
print(x+y)

#연결연산자
#문자열일 경우에는 단순히 "연결"만 해줌
x="7"
y="6"
print(x+y)

#변수의 이름규칙
#1. 대문자와 소문자를 구별한다
x = 100
X = 200
print(x,X)

#2. 중간공백은 불가능함, 띄어쓰기 대신에 "_"를 사용
variable_1 = 30
print(variable_1)

#3. 숫자먼저오면 안됨(나중에 숫자 오는 것은 가능함)
#e.g. 1x,2x ...

#4. 특수문자는 불가능함

#많이쓰는 변수 이름 방법 : 첫글자는 소문자로 나머지는 단어는 대문자로
#ex
myNewCar = "sonata"
print(myNewCar)

#변수의 초기화
score = 10
print("id of score = 10 : {}".format(id(score)))
score = score + 1
print("id of score = score + 1 : {}".format(id(score)))

#ex
x = 100
y= 200
sum = x+y
print(x,"과","y의 합은",sum,"입니다.") #문자열에는 ""쌍따옴표 붙는것에 유의

#input
#외부의 입력장치로부터 입력을 받을때 사용하는 함수
d = input()
print(d)

#두 정수를 입력받아서 합을 출력하는 프로그램 작성하기
x = int(input("첫 번째 정수를 입력하세요")) #int를 붙여줘야 함에 유의,input의 return은 string임
y = int(input("두 번째 정수를 입력하세요")) 
sum = x+y
print(x,"과 ",y,"의 합은 ",sum,"입니다.")

d="ds"
print(type(d))