# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

time_de_noise = [35.026, 65.670, 119.359, 210.036, 241.289, 238.413, 253.778, 241.353, 257.323, 228.83]

plt.plot([(x + 1) * 10 for x in range(10)], time_de_noise, 'v-')
# x轴范围,y轴范围
plt.axis([10, 100, 0, 300])
# y轴文字
plt.ylabel(u'Time per request(m)')
# x轴文字
plt.xlabel(u'Number of concurrency')
# 标识
plt.legend(loc=2, ncol=3)
# 展示
plt.show()
