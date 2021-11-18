count = 0
arr = [1, 13, 7, 3, 25, 11, 9]


def beEven(cur, n):
    global arr
    global count

    if (cur + 1 == n):
        count += 1
        return

    lval = 0

    for i in range(cur, n):
        lval += arr[i]

        # 짝수 이면 다음 홀수 그룹을 찾는다.
        if (lval % 2 == 0):
            # i+1부터 그룹을 찾는다.
            beOdd(i + 1, n)


def beOdd(cur, n):
    global count

    if (cur == n):
        count += 1
        return

    lval = 0

    for i in range(cur, n):
        lval += arr[i]

        # 홀수 이면 다음 짝수 그룹을 찾는다.
        if (lval % 2 != 0):
            # i+1 부터 그룹을 찾는다.
            beEven(i + 1, n)


if __name__ == '__main__':
    beEven(0, len(arr))
    print(count)