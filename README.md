# -Prediction-of-the-ships-destination-port
 The first-place solution in the algorithm model track of the 3rd "Digital Intelligence Port and Shipping" Data Innovation Application Competition
predict_last_des.py负责对数据进行处理，制作标签，最终生成数据集。原始数据集在data中，处理后的数据集保存在data\temp中。
model.py负责对结果进行预测，将预测结果保存为result.csv文件
temp文件夹负责保存编码器的拟合文件（label_encoder.pkl）；模型权重（AutogluonModels文件夹）；预测结果(result.csv)和处理后的数据集(***set_new.csv)等
模型：使用Autogluon模型，该模型集成了xgboost，lightgbm等经典机器学习模型
启动命令：python predict_last_des.py && python model.py
