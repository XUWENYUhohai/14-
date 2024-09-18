# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
 
# https://blog.csdn.net/Guo_Python/article/details/105820358

 
def draw_pr(confidence_scores, data_labels):
    plt.figure()
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
 
    #精确率，召回率，阈值
    precision,recall,thresholds = precision_recall_curve(data_labels,confidence_scores)
    print(precision)
    print(recall)
    print(thresholds)
 
    from sklearn.metrics import average_precision_score
    AP = average_precision_score(data_labels, confidence_scores) # 计算AP（面积）
    plt.plot(recall, precision, label = 'pr_curve(AP=%0.2f)' % AP)
    plt.legend()#显示label内容
    plt.show()
 
 
if __name__ == '__main__':
    # 正样本的置信度,即模型识别成１的概率
    confidence_scores = np.array([0.9, 0.78, 0.6, 0.46, 0.4, 0.37, 0.2, 0.16])
    # 真实标签
    data_labels = np.array([1,1,0,1,0,0,1,1])
 
    draw_pr(confidence_scores, data_labels)
