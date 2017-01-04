# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

time_pearson = [80.539, 129.841, 156.452, 223.789, 274.803, 318.888, 362.025, 438.480, 434.690, 472.400]

plt.plot([(x + 1) * 10 for x in range(10)], time_pearson, 'v-')
# x轴范围,y轴范围
plt.axis([10, 100, 0, 500])
# y轴文字
plt.ylabel(u'Time per request(m)')
# x轴文字
plt.xlabel(u'Number of concurrency')
# 标识
plt.legend(loc=2, ncol=3)
# 展示
plt.show()
