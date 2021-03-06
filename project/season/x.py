#!/user/bin/env python
#!-*-coding:utf-8 -*-

list_remain = ['名称', '报告期', '最新评级', '短期债务/总债务',
               '货币资金/短期债务', '净利润(亿元)', '筹资活动现金流(亿元)',
               '主营业务利润率(%)', '投资活动现金流(亿元)',
               '净资产回报率(%)', '总资产报酬率(%)', '流动比率',
               '资产负债率', '经营活动现金流(亿元)'
               '经营活动现金流/短期债务', '可供投资现金流(亿元)']

weight = [('短期债务/总债务', 0.0569), ('主营业务收入(亿元)', 0.0483), ('流动比率', 0.0456),
          ('筹资活动现金流(亿元)', 0.0451), ('主营业务收入增长率(%)', 0.0451), ('存货周转率', 0.0424),
          ('非筹资活动现金流(亿元)', 0.0392), ('净资产回报率(%)', 0.037), ('资产负债率', 0.0365),
          ('经营活动现金流/短期债务', 0.0359), ('货币资金/短期债务', 0.0354), ('速动比率', 0.0343),
          ('净资产(亿元)', 0.0338), ('投资活动现金流(亿元)', 0.0338), ('带息债务/总投入资本', 0.0322),
          ('货币资金/总债务', 0.0317), ('货币资产(亿元)', 0.0311), ('总现金流(亿元)', 0.03),
          ('净利润/带息债务', 0.0295), ('可供投资现金流(亿元)', 0.0284), ('总资产报酬率(%)', 0.0274),
          ('净利润(亿元)', 0.0252), ('经营活动现金流(亿元)', 0.0231), ('主营业务利润率(%)', 0.022),
          ('净债务(亿元)', 0.0209), ('主营业务利润(亿元)', 0.0204), ('总债务(亿元)', 0.0193),
          ('企业性质_民营企业', 0.0193), ('总资产(亿元)', 0.0156), ('带息债务(亿元)', 0.0139),
          ('二级分类_其他', 0.0059), ('企业性质_地方国有企业', 0.0054), ('rank_by_p2', 0.0048),
          ('企业性质_其他', 0.0032), ('企业性质_中央国有企业', 0.0027), ('whether_h/d', 0.0021),
          ('rank_by_p3', 0.0021), ('是否上市_否', 0.0021), ('二级分类_零售', 0.0021), ('rank_by_p4', 0.0016),
          ('总资产(亿元)_1.0', 0.0016), ('是否上市_是', 0.0016), ('whether_chouzi', 0.0011),
          ('二级分类_节能服务', 0.0011), ('whether_su', 0.0005), ('总资产(亿元)_4.0', 0.0005),
          ('二级分类_信息技术', 0.0005), ('二级分类_文化传媒', 0.0005), ('二级分类_日用品', 0.0005),
          ('二级分类_物流快递', 0.0005), ('n_loss', 0.0), ('whether_jin', 0.0), ('whether_liu', 0.0),
          ('whether_d/a', 0.0), ('rank_by_p1', 0.0), ('总资产(亿元)_-1.0', 0.0), ('总资产(亿元)_0.0', 0.0),
          ('总资产(亿元)_2.0', 0.0), ('总资产(亿元)_3.0', 0.0), ('是否交通', 0.0), ('二级分类_建筑业', 0.0),
          ('二级分类_电气', 0.0), ('二级分类_石油天然气', 0.0)]

sorted_remain = [i[0] for i in weight if i[0] in list_remain]
