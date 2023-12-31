## 数据说明：

train_data_sample.csv和dev_data_sample.csv是从train_data.csv和dev_data.csv抽取出来的部分数据样例。

train_data.csv：选手用于进行模型训练的数据集，包含天气信息和光伏工作数据，时间周期为2023-01-01至2023-08-27，数据采集频率为每分钟一条，共计344134条数据（有一些日期存在某些时刻数据缺失，不足1440条），选手需要根据这些数据，训练可预测未来15min、未来30min、未来1h、未来4h、未来24h 共计5个时间点光伏产生的有功功率。

dev_data.csv：对选手的模型进行效果验证的数据集，包含天气信息和光伏工作数据（剔除了光伏的有功功率），时间周期为2023-08-28至2023-08-30，数据采集频率为每分钟一条，共计 3\*1440=4320 条数据，选手需要根据这些数据，利用训练好的模型，预测这些时间点未来15min、未来30min、未来1h、未来4h、未来24h光伏产生的有功功率。

result_sample：结果样例，选手需要按照样例的格式，把对应时间点的未来某个时间点（15min、30min、1h、4h、24h）的光伏有功功率预测结果填在相应的位置。time表示验证集中给出的验证数据时间点，15min表示该时间点未来第15分钟（time=2023-08-28 00:00，则15min对应的时间点为2023-08-28 00:15）预测到的光伏有功功率，30min、1h、4h、24h以此类推。

部分数据字段说明：
* 光伏电池板1 2 3 4线路电压电流：PV1电压、PV1电流、PV2电压、P2电流、PV3电压、PV3电流、PV4电压、PV4电流
* 光伏收集的电功率：输入功率
* 三项电压电流：Ua、Ub、Uc、Ia、Ib、Ic

模型评分标准：各个任务按照准确率取平均。