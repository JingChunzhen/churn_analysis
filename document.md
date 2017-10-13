Title         : 基于XGBoost和随机森林的流失分析
Author        : 景春臻 levelup.ai
Logo          : False

[TITLE]

# 狂暴之翼游戏数据分析 

* 游戏玩家总人数 17965
* 动作总数 4867
* 游戏中VIP的用户人数 151
* 游戏玩家的流失情况 60%会流失
* 非VIP玩家的流失情况 60%会流失
* VIP玩家的流失情况 6%会流失
* 流失用户的动作序列长度的平均值 277
* 非流失用户的动作序列长度的平均值 2831

# 数据预处理

* 数据泄露问题的处理：流失玩家和非流失玩家之间的动作序列长度差别较大，可能会造成数据泄露问题，使用TF-IDF作为特征解决数据泄露问题
* 数据不平衡问题的处理：游戏玩家中大约会有60%会流失，对非流失玩家进行采样，得到的流失玩家和非流失玩家的比例为7348:7142，
比例大约为1:1

# 实验前思考的问题

* 游戏中VIP玩家由于流失比例仅为6%，所以动作序列中的VIP动作是否会对结果造成较大的影响
* 如何评判一个模型的效果，如果只用一个模型是否具备说服力

# 模型的选择

流失分析实质上是一个二分类任务，可使用多种模型进行实验，最终选择集成学习中的Boosting方法和Bagging方法来进行，XGBoost对损失函数进行了二阶泰勒展
开，分类性能优异，随机森林由于其各个基分类器在样本和属性上的多样性，最终集成的泛化性能较好,机器学习框架使用[sklearn](http://scikit-learn.org/)

## XGBoost

使用网格搜索和10折交叉验证对XGBoost进行参数的优化，并使用early-stopping防止模型过拟合，核心代码与模型参数如下

```python
model = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=100, max_depth=15, subsample=1)
eval_set = [(X_validate, Y_validate)]
model.fit(X_train, Y_train, early_stopping_rounds=20,
            eval_metric="logloss", eval_set=eval_set, verbose=True)
```

使用XGBoost得到的结果如下

* 在测试集上的准确率：86.79%
* 对非流失玩家预测的精确率：92.72%
* 对非流失玩家预测的召回率：79.70%
* 在测试集上非流失玩家的支持度：2400
* 对流失玩家预测的精确率：82.36%
* 对流失玩家预测的召回率：93.81%
* 在测试集上流失玩家的支持度：2424

所提取的重要动作（前20个）信息如下

动作 | 动作说明 | 重要性
-----|---------|-------
ui_bigmap_indexnum/im	| 战役/点击关卡 |	0.0330210775
ui_city_lefttop/mc_window/maintask/btngoto1	| |	0.0283372365
requesting/instance2	| 网络连接中/点击屏幕 	| 0.0238875877
uitop_talk/bg	| 主城/对话点击 |	0.0210772827
/mc_window/mc_window/kktishi	| 主城/引导点击右下角图标时点击别处 |	0.0206089001
uidialog_target/lingquBtn	| 主城/主线或支线-领取奖励 |	0.0203747079
uidialog_victory/gotocity	| 关卡/关卡结束后-返回主城 |	0.0192037467
uidialog_backpack/btnClose |	背包/关闭 |	0.0166276339
uidialog_guanqiaxuanze/go	| 战役/关卡进入 |	0.0161592513
ui_city_rightbottom/mc_window/mc_window/mgr	| 主城/右下方收缩按钮 |  	0.0161592513
uidialog_club_tip/contentMc/bigBtn	  | |	0.0161592513
ui_city_leftbottom/mc_window/chatMessageMc/bg	| 主城/点击聊天栏 | 	0.0156908669
ui_city_centerbottom/btn	| 主城/点击经验条 |	0.0154566746
ui_shade_center/bg	| 关卡/关卡内暂停-点击其他地方 |	0.0149882901
uidialog_bosschest/btnClose |	关卡/结束后，点击继续时点击屏幕 	| 0.0147540979
ui_city_rightbottom/mc_window/mc_window/newequip |	主城/快速穿戴装备 |	0.0145199066
uitop_hand/kktishi7	| 套装合成/引导关闭界面是点击别处 |	0.0142857144
ui_city_righttop/mc_window/bigmap	| 主城/战役 |	0.0138173299
uidialog_target/btngo	| 主线/支线前往 | 	0.0131147541
uidialog_lingqu_tomorro/btn_lingqu	| 明日领取/领取 |	0.0128805619
/list/icontent/c1/getbtngetbtn	| |	0.0124121783
## 随机森林

使用随机森林得到的结果如下

* 在测试集上的准确率：84.57%
* 对非流失玩家预测的精确率：86.54%
* 对非流失玩家预测的召回率：82.38%
* 在测试集上非流失玩家的支持度：2400
* 对流失玩家预测的精确率：81.70%
* 对流失玩家预测的召回率：87.1%
* 在测试集上流失玩家的支持度：2424

所提取的重要动作（前20个）信息如下

动作 | 动作说明 | 重要性
-----|---------|----------
/blist/icontent/c2/btnbtn	| |	0.0230295461
uidialog_10l/mc_model/cost_diamond_one_btn	| |	0.0145689162
uidialog_shenqi/bg/btnClose	| |	0.0133559941
ui_city_righttop/mc_window/tomorrow/btn	| 主城/明日领取 | 	0.0123999483
uidialog_equip/window/jinjieMC/btnSure	| 锻造/精炼 | 	0.0119780654
uidialog_wing_detail/tab3	| 怒翼/任一怒翼-组合页签 |	0.0116050021
/mc_window/mc_window/subitems/sub2	| 主城/王者殿堂 |	0.0110919308
uidialog_equipshow/mc_hecheng/btn_do	| 套装合成/合成 |	0.0103155078
ui_bigmap_righttop/mc_window/homeBtn	| 战役/回城 |	0.0096913018
uidialog_10lbx/showbg/btn	| 命运宝藏/钻石一次抽到装备 |	0.0091626334
uidialog_equip/closeBtn	| 锻造/关闭 |	0.0089656248
ui_battle_auto/btnFightMode	| 竞技场/自动战斗 |	0.008852379
mc_duanzaoBottom/btnOnekeySure |	锻造/一键强化 |	0.0084886179
uidialog_club_tip/contentMc/bigBtn	  | |	0.0083878282
uidialog_mbtask/closeBtn	| |	0.0081743806
uidialog_equipshow/close	| 套装/关闭 |	0.0081284633
uidialog_shilian/enterBtn	| 永恒之塔/进入/领取宝箱 | 	0.0080005634
/listMC/icontent/c1/btnbtn	| |	0.0079048626
uidialog_victory/yindao_xingxing | 关卡/第一次星星引导界面 |	0.0078938585
uidialog_victory/nextmission	| 关卡/精英关卡选择关卡 | 0.0078371413
uidialog_guide_tip/btnOK	| 战斗中出现的提示面板/点击【明白了】按钮 | 	0.0077829487


# 模型的评价和仍存在的问题

* 在模型所提取的动作中VIP动作（即动作信息中包含‘vip’的动作）并不多，表示VIP玩家对于流失分析的影响没有很大，VIP玩家数量只有151个，
总玩家数量为17965个
* 两个模型所提取的动作信息会出现不一致的情况，在两个模型的前20个，50个，100个， 300个重要动作中，共同的动作个数分别为1个
，22个，46个，225个, 而两个模型的重要动作重合性越高，预测结果越可信
* 为了更好的在策划上解决问题，模型应该提高对流失玩家预测的召回率，当前XGBoost所做的针对流失玩家的召回率为93.81%, 还应该继续提高
* 游戏流失问题的产生，可能会和游戏在设计，编写环节中出现的bug有关，这点需要配合时序关联分析来进行
* 影响玩家流失与否的因素不仅仅包含动作信息，还包含游戏的题材等信息，即玩家是否是该游戏的受众用户，这点使用模型来单纯的在动作序列上进
行预测较为困难
