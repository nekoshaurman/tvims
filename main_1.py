import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import skew, kurtosis, poisson, norm, chi2
import math
from sympy import *
import statsmodels.api as sm

def fac(n):
    if n == 1:
        return 1
    return fac(n - 1) * n


def toFixed(numObj, digits=4):
    return f"{numObj:.{digits}f}"


def laplace(x):
    return norm.cdf(x)


a = 0.00
a1 = 0.02
b = 1.6
lambda0 = 3.0
lambda1 = 1.0

arr1 = np.array([
                1, 1, 1, 0, 4, 4, 1, 0, 1, 0, 0, 0, 2, 1, 0, 2,
                2, 3, 1, 0, 0, 0, 1, 2, 1, 0, 0, 2, 0, 1, 1, 1,
                0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 0, 2, 0, 2, 0, 2,
                0, 2
])

arr1.sort()

datastandart = [0]*50
for i in range(len(datastandart)):
    datastandart[i] = arr1[i]
datastandart = np.array(datastandart)


print("Задание a)")
n = np.unique(arr1).size
print("Вариационный ряд: ", *arr1)
print("Размер выборки: ", arr1.size)


taker = [0] * n
for i in range(np.unique(arr1).size):
    taker[i] = arr1[arr1 == np.unique(arr1)[i]].size / arr1.size
print("Частоты: ", *taker, "\n")


# Строим эмпирическую функцию
ecdf = sm.distributions.ECDF(arr1)
x = np.linspace(np.min(arr1), np.max(arr1))
y = ecdf(x)
plt.step(x, y)
plt.title('Эмпирическая функция')
plt.xlabel('Значения')
plt.ylabel('Вероятность')
plt.show()

# Строим гистограмму
plt.hist(arr1, bins=np.arange(np.max(arr1)+2)-0.5, edgecolor='black')
plt.title('Гистограмма')
plt.xlabel('Значения')
plt.ylabel('Частоты')
plt.show()



print("Задание b)")
Vib_sred = np.mean(arr1)
print("Выборочное среднее (I): ", Vib_sred)
# (ii)
Vib_disp = np.var(arr1)
print("Выборочная дисперсия (II): ", toFixed(Vib_disp))
# (iii)
median = np.median(arr1)
print("Медиана (III): ", median)
# (iV)
skewness = skew(datastandart)
print("Ассиметрия (IV): ", skewness)
# (V)
excess = kurtosis(datastandart)
print("Эксцесса (V): ", excess)
# (VI)
Prob1 = ((arr1 >= a) & (arr1 <= b)).sum() / arr1.size
print("Вероятность (VI) X:[a, b] ", toFixed(Prob1), "\n")



print("Задание c)")
# По методу моментов
lambda_moment = np.mean(arr1)
print("По методу моментов: ", toFixed(lambda_moment))
lambda_max = np.mean(arr1)
print("Оценка максимального правдоподобия: ",toFixed(lambda_max), "\n")



print("Задание d)")
z_alpha_2 = norm.ppf(1 - a1/2)
print("Из функции Лапласа: ", z_alpha_2)

x = symbols('lambda')
print("Доверительный интервал: ", solve((x - np.mean(arr1))**2 < (z_alpha_2**2 * x) / len(arr1)), "\n")



print("Задание e)")
# Строим гистограмму частот
hist, bin_edges = np.histogram(arr1, bins=np.arange(-0.5, 5.5, 1))

# Частоты
expected_freq = len(arr1) * poisson.pmf(np.arange(0, 5), lambda0)

# Статистика критерия X^2
chi2_stat = np.sum((hist - expected_freq)**2 / expected_freq)

# Критическое значение X^2
df = len(hist) - 1
chi2_crit = poisson.ppf(1 - a1, df) * df

print("Статистика критерия: ", chi2_stat)
print("Критическое значение: ", chi2_crit)

# Результат
if chi2_stat > chi2_crit:
    print("Гипотеза отвергается на уровне значимости", a1)
else:
    print("Гипотеза не отвергается на уровне значимости", a1)

p_value = 1 - poisson.cdf(chi2_stat / df, df)
print("Наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть гипотезу:", p_value, "\n")



print("Задание f)")
# Строим гистограмму частот
_, count = np.unique(arr1, return_counts=True)

# Частоты
expected_freq = len(arr1) * poisson.pmf(np.arange(0, 5), np.mean(arr1))

# Статистика критерия X^2
chi2_stat = np.sum((count - expected_freq) ** 2 / expected_freq)

# Критическое значение X^2
df = len(count) - 1
chi2_crit = poisson.ppf(1 - a1, df) * df

print("Статистика критерия: ", chi2_stat)
print("Критическое значение: ", chi2_crit)

# Результат
if chi2_stat > chi2_crit:
    print("Гипотеза отвергается на уровне значимости", a1)
else:
    print("Гипотеза не отвергается на уровне значимости", a1)

p_value = 1 - poisson.cdf(chi2_stat / df, df)
print("Наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть гипотезу:", p_value, "\n")



print("Задание g)")
# Не готово
