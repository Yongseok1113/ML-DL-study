import time

# case1 = "1 13 7 3 25 11 9"
'''
    정답: 6개
    | 1 13 | 7 3 25 11 9 |
    | 1 13 | 7 | 3 25 11 9 |
    | 1 13 | 7 3 25 | 11 9 |
    | 1 13 17 3 | 25 11 9 |
    | 1 13 17 3 | 25 | 11 9 |
    | 1 13 17 3 25 11 | 9 |
    총 6 개
'''
# case3 = "84 87 78 16 94 36 87 93 50 22"
'''
    정답: 3개
    | 84 | 87 78 16 94 36 87 93 50 22 |
    | 84 | 87 78 16 94 36 87 93 | 50 22 |
    | 84 87 78 16 94 36 87 | 93 50 22 |
'''

"""
    first_dict, next_dict 는 입력된 1차 배열에서 그룹합이 짝수가 되는 인덱스를 키 값으로 가지며, 해당 경우에 그룹합이 홀수가 되는 인덱스의
    리스트를 값으로 가집니다.
    첫 번째 짝수그룹, 홀수그룹은 반드시 존재해야하기 때문에 고정으로 구한 후 
    다음 next_dict(딕셔너리)의 짝수그룹 인덱스, 홀수 그룹 인덱스를 구합니다.
    이 과정에서 각 경우의 수에 대해 남아있는 나머지 수를 더했을때 번갈아 나오는 조건이면 개수를 세고 아니면 무시하는 방법으로 구했습니다.
"""
answer = 0
# 입력
case = str(input())

# 시간 측정 시작
start_time = time.perf_counter()
input_array = list(map(lambda x: int(x), case.split(' ')))
odd_index_list = [idx for idx, e in enumerate(input_array) if e % 2]

first_dict = {}
# 첫 짝수 모든 경우 준비
for i in range(odd_index_list[-1]):
    if not sum(input_array[:i+1]) % 2:
        first_dict[i] = []

# 첫 홀수 모든 경우 준비
for even_idx in first_dict.keys():
    for odd_idx in odd_index_list:
        if odd_idx > even_idx:
            check = sum(input_array[even_idx+1:odd_idx+1])
            if check % 2:
                first_dict[even_idx].append(odd_idx)


def loop(temp_dict):
    next_dict = {}
    num_cases = 0
    make_next = False
    for temp_even_idx, odd_list_idx in temp_dict.items():

        # if sum(input_array[temp_even_idx + 1:]) % 2:
        #     num_cases += 1

        for temp_odd_idx in odd_list_idx:
            # 짝수 이후 모든 값 합이 홀수면 증가
            if temp_odd_idx == len(input_array) - 1:
                num_cases += 1
            # 홀수 이후 모든 값 합이 짝수면 증가
            check = input_array[temp_odd_idx+1:]
            if check and not sum(check) % 2:
                num_cases += 1

            # 최소 첫째 짝, 홀 그룹 상태에서 구한 경우의수가 이후 경우의 수와 겹치기 때문에 중복 탐색을 막기위해
            # 한번만 수행합니다. make_next 를 주석처리해도 결과는 같습니다.
            if not make_next:
                # 다음 짝 생성
                for i in range(temp_odd_idx+1, len(input_array)):
                    if not sum(input_array[temp_odd_idx+1:i + 1]) % 2:
                        next_dict[i] = []

                # 다음 홀 생성
                for next_even_idx in next_dict.keys():
                    for odd_idx in odd_index_list:
                        if odd_idx > next_even_idx:
                            check = sum(input_array[next_even_idx + 1:odd_idx + 1])
                            if check % 2:
                                next_dict[next_even_idx].append(odd_idx)
                make_next = True
    return num_cases, next_dict


num_cases, next_dict = loop(first_dict)
answer += num_cases

while True:
    num_cases, next_dict = loop(next_dict)
    answer += num_cases

    if not next_dict:
        break
end_time = time.perf_counter()

print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")
print("answer: ", answer)










