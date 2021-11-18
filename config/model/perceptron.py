import numpy as np


def AND(x1, x2):
    """
        theta는 뉴런을 활성화 시키는 임계값

    :param x1: 입력1
    :param x2: 입력2
    :return: 활성 여부 반환(0 또는 1)
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND_np(x1, x2):
    """
        가중치는 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수
        편향은 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조정하는 매개변수
        # 편향이라는 용어는 '한쪽으로 치우쳐 균형을 깬다' 라는 의미, 실제로 입력이 모두 0이어도 결과는 편향값(-0.7)
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.7
    # 넘파이 배열끼리 곱셈은 원소수가 같을경우 원소별 곱
    tmp = np.sum(w*x) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def OR_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.4

    tmp = np.sum(x*w) + bias
    if tmp >= 0:
        return 1
    else:
        return 0


def NAND_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    bias = 0.7

    tmp = np.sum(x*w) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR_np(x1, x2):
    return AND_np(OR_np(x1, x2), NAND_np(x1, x2))


if __name__ == "__main__":
    x1, x2 = (0, 1)
    print(XOR_np(x1, x2))






