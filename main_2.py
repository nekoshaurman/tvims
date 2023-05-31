import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, poisson, norm, chi2
import math
from scipy import stats


def fac(n):
    if n == 1:
        return 1
    return fac(n - 1) * n


def toFixed(numObj, digits=4):
    return f"{numObj:.{digits}f}"


def laplace(x):
    return norm.cdf(x)

alpha = 0.10
b = 1.54
c = -4.00
d = 1.00
h = 2.00
a0 = -3.00
sigma0 = 5.00
a1 = -19.00
sigma1 = 5.00

arr1 = np.array([2.58, 4.98, 8.56, 7.46, -2.35, -2.53, -4.55, 5.66, -0.23, -4.14,
                 2.80, 7.96, 3.67, 4.80, 9.16, -3.24, 8.90, -3.85, -0.75, -3.82,
                 5.72, -1.69, -2.33, -2.23, 0.42, -1.40, 3.24, -7.38, 6.87, -5.09,
                 -5.38, 0.08, -3.48, -1.54, -4.51, 3.39, 0.82, -1.83, -2.03, 12.39,
                 -3.69, 3.32, 4.11, 3.05, 1.64, 8.64, -2.74, 7.30, -10.23, 3.12])

arr1.sort()
data = np.unique(arr1)

datastandart = [0]*50
for i in range(len(datastandart)):
    datastandart[i] = arr1[i]
datastandart = np.array(datastandart)

print("Задание a)")
n = data.size
print("Вариационный ряд: ", *arr1)
print(arr1.size)


taker = [0] * n
for i in range(data.size):
    taker[i] = arr1[arr1 == data[i]].size / arr1.size
print(taker)


'''
plt.step(np.sort(arr1), np.arange(1, len(arr1)+1)/len(arr1))

plt.title('Эмпирическая функция распределения')
plt.xlabel('Значение')
plt.ylabel('Вероятность')
plt.show()

plt.hist(arr1, bins=np.arange(arr1.min(), arr1.max()+1)-0.5, rwidth=0.7)

# Добавляем заголовок и метки осей
plt.title('Гистограмма вариационного ряда')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.xticks([-12,-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.show()


hist, bin_edges = np.histogram(arr1, bins=10)
hist = hist / hist.sum()
intervals = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.plot(intervals, hist, color='black', linestyle='-', linewidth=2)
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Полигон частот')
plt.xticks([-12,-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.show()
'''

arr2 = np.zeros(arr1.size)



print("\nЗадание b)")
#I
Vib_sred = np.mean(arr1)
print("Выборочное среднее (I): ", Vib_sred)
#II
Vib_disp = np.var(arr1)
print("Выборочная дисперсия (II): ", toFixed(Vib_disp))
#III
median = np.median(arr1)
print("Медиана (III): ", median)
#IV
skewness = skew(datastandart)
print("Ассиметрия (IV): ", skewness)
#V
excess = kurtosis(datastandart)
print("Эксцесса (V): ", excess)
#VI
Prob1 = (arr1.size - arr1[arr1 <= c].size - arr1[arr1 >= d].size) / arr1.size
print("Вероятность X:[a, b] ", toFixed(Prob1))



print("\nЗадание c)")
# По методу моментов
alpha_moment = np.mean(arr1)
print("По методу моментов a: ", toFixed(alpha_moment))
sigma_moment = np.var(arr1)
print("По методу моментов s^2: ", toFixed(sigma_moment))

alpha_moment = np.mean(arr1)
print("Максимальное правдоподобие a: ", toFixed(alpha_moment))
sigma_moment = np.var(arr1)
print("Максимальное правдоподобие s^2: ", toFixed(sigma_moment))



print("\nЗадание d)")
# Вычисляем выборочное среднее и выборочную дисперсию
x_bar = np.mean(arr1)
s2 = np.var(arr1, ddof=1)
# Находим критические значения t-распределения Стьюдента и хи-квадрат распределения
t_alpha_2 = stats.t.ppf(1 - alpha/2, df=len(arr1)-1)
chi2_alpha_2 = stats.chi2.ppf(1 - alpha/2, df=len(arr1)-1)
# Находим доверительные интервалы для параметров a и S^2
a_ci = (x_bar - t_alpha_2 * np.sqrt(s2/len(arr1)), x_bar + t_alpha_2 * np.sqrt(s2/len(arr1)))
s2_ci = ((len(arr1)-1)*s2/chi2_alpha_2, (len(arr1)-1)*s2/stats.chi2.ppf(alpha/2, df=len(arr1)-1))
print("Доверительный интервал для параметра a: ", a_ci)
print("Доверительный интервал для параметра S^2: ", s2_ci)



print("\nЗадание e)")
# Параметры нормального распределения
a = -3
s2 = 25
# Разбиваем выборку на k интервалов
k = int(np.ceil(1 + np.log2(len(arr1))))
# Находим границы интервалов
bins = np.linspace(np.min(arr1), np.max(arr1), k+1)
# Вычисляем теоретические частоты в каждом интервале
f_exp = np.array([stats.norm.cdf(bins[i+1], loc=a, scale=np.sqrt(s2)) - stats.norm.cdf(bins[i], loc=a, scale=np.sqrt(s2)) for i in range(k)])
# Вычисляем наблюдаемые частоты в каждом интервале
f_obs, _ = np.histogram(arr1, bins=bins)
# Вычисляем статистику критерия X^2
chi2 = np.sum((f_obs - f_exp * len(arr1))**2 / (f_exp * len(arr1)))
# Находим критическое значение X^2 распределения с k-1 степенями свободы для уровня значимости alpha = 0.10
chi2_crit = stats.chi2.ppf(1 - alpha, df=k-1)
# Сравниваем статистику критерия X^2 со значением критической области и делаем вывод о гипотезе
if chi2 > chi2_crit:
    print("Гипотеза о согласии с нормальным распределением отвергается на уровне значимости alpha = 0.10")
else:
    print("Гипотеза о согласии с нормальным распределением не отвергается на уровне значимости alpha = 0.10")
# Находим наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть данную гипотезу
alpha_max = 1 - stats.chi2.cdf(chi2, df=k-1)
print("Наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть гипотезу: ", alpha_max)



print("\nЗадание f)")
# Разбиваем выборку на k интервалов
k = int(np.ceil(1 + 3.322 * np.log10(len(arr1))))
# Находим границы интервалов
bins = np.linspace(np.min(arr1), np.max(arr1), k+1)
# Оцениваем параметры нормального распределения по выборке
a = np.mean(arr1)
s2 = np.var(arr1, ddof=1)
# Вычисляем теоретические частоты в каждом интервале
f_exp = np.array([stats.norm.cdf(bins[i+1], loc=a, scale=np.sqrt(s2)) - stats.norm.cdf(bins[i], loc=a, scale=np.sqrt(s2)) for i in range(k)])
# Вычисляем наблюдаемые частоты в каждом интервале
f_obs, _ = np.histogram(arr1, bins=bins)
# Вычисляем статистику критерия X^2
chi2 = np.sum((f_obs - f_exp * len(arr1))**2 / (f_exp * len(arr1)))
# Находим критическое значение X^2 распределения с k-1 степенями свободы для уровня значимости alpha = 0.01
chi2_crit = stats.chi2.ppf(1 - alpha, df=k-1)
# Сравниваем статистику критерия X^2 со значением критической области и делаем вывод о гипотезе
if chi2 > chi2_crit:
    print("Гипотеза о согласии с нормальным распределением отвергается на уровне значимости alpha = 0.10")
else:
    print("Гипотеза о согласии с нормальным распределением не отвергается на уровне значимости alpha = 0.10")
# Находим наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть данную гипотезу
alpha_max = 1 - stats.chi2.cdf(chi2, df=k-1)
print("Наибольшее значение уровня значимости, на котором еще нет оснований отвергнуть гипотезу: ", alpha_max)


