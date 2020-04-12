# ALI_COMPETITION
    本赛题基于位置数据对海上目标进行智能识别和作业行为分析，要求选手通过分析渔船北斗设备位置数据，得出该船的生产作业行为
    ，具体判断出是拖网作业、围网作业还是流刺网作业。初赛将提供11000条(其中7000条训练数据、2000条testA、2000条testB)渔船
    轨迹北斗数据。下面是数据示例：
![](https://github.com/791092214/ALI_COMPETITION/raw/master/1586681538(1).png)
    
# 数据预处理
    从上面的数据示例可以看到，时间这个属性是由日期和具体的时间点连在一起的，这个是无法导入模型之中的。所以这里采取的方案是将
    time转换成秒，比如：0921 01：00，这里将01:00变成1*60*60秒，这代表了01:00这个时刻，是一天中的第3600秒。然后将date信息直
    接转化成float类型的数据，比如，将0921直接转换成数字921。
    
    在将时间信息处理完之后，就直接将所有的数据进行的标准化处理。在复盘的时候，发现其实这个地方有做的不好的地方。因为我
    没有处理异常值。当时的考虑是，因为待预测的数据集中也有异常值，而且我相信神经网络强大的解释能力，能够把待预测的数据中
    的异常值也能判断出来。现在想来，这种做法是不太对的。首先，神经网络对异常值本身就是敏感的。其次，至少也应该尝试下去除
    异常值的方法。

# 建模
    由于这个识别渔船类型的任务是比较特别的，网上也没有很合适的神经网络模型，所以采取自己构建神经网络。这就是一个漫长的
    过程了。只能一层一层的增加，每层的数量一点点的变化，不断的尝试，最后得到了现在的模型。

# 最后的处理
    因为每个ID都有差不多300条数据，这些数据代表了一艘渔船在一天中的不同时间点的，方位，速度，方向。而我构建的模型却是把每
    一条数据当成单独的数据来导入模型，然后训练，并预测结果的。也就是说，最终，针对一个待预测的ID，最终将会有300条预测的
    结果。而这些结果大多数情况下都不是完全相同的。比如，对ID是1的渔船的300条数据进行预测后，可能会有200条是拖网，50条是
    围网，50条是刺网。这个时候当然取类别最多的那个类型当做是最终的预测结果。然而，在实际的处理过程中，发现模型对单条数据
    的预测结果的ACC可以达到0.9，可是最终的预测效果却不好。比如：训练集一共20个ID，有 20*300 条数据。虽然训练集的ACC可以
    到0.9，但是最终的预测结果的F1-SCORE只有0.8。经过分析，觉得可能是没有预测对的数据，最大可能性的造成的对ID判断的错误。
    比如：这里有10个ID，10*300 条数据，尽管这3000条数据中，我预测对了90%，但是那没有预测对的10%都均匀的分布在各个ID的结
    果中，从而导致了对最终结果判断的错误。比如：对ID是2的300条数据进行预测，结果是，拖网103条，围网100条，刺网97条。而正
    确结果可能是围网。但是由于模型刚好就把那么几条数据预测成了拖网，这就导致了对这个ID预测的错误。而我的这个，最后的处理，
    就是当碰到像，拖网：103，围网：100，刺网97，这类型的均衡局面的时候， 就直接选择第二多的类型当做最终结果。那么，如何界定
    这个均衡局面，我采取的方案是，构建一个比例，即，出现次数最多的种类/300,即出现次数最多的类型的占比。当然，这个比例是需
    要不断节，尝试，以求达到最好效果的。
    
    在后面复盘整个比赛的时候，针对，最后的处理，这一部份，我发现我也犯了一个错误，那就是，没有构建验证集。当时因为CPU的算
    力确实不能满足需求，所以是第一次尝试使用tensorflow框架来进行GPU加速，还不太熟悉这个东西。但总的来说，没有构建验证集而只
    依赖训练集的结果，这无论如何都是一个巨大的失误。
