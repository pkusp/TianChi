## eval results
## 2017/5/27
## cyyang


import os
import datetime

import numpy as np
import pandas as pd



class EvalResults:
    ''' eval results '''
    def __init__(self, test_set_start_day_id):
        self.test_set_start_day_id = test_set_start_day_id
        return


    def read_data_and_eval(self, filename_all, filename_prdt):
        print('读全体测试数据')
        # day_id,sale_nbr,buy_nbr,cnt,round
        # 65,O7211,PAX,33,25000
        # 45,O1164,O2062,1,200
        df_all = pd.read_csv(filename_all, comment='#')
        df_train = df_all.loc[(df_all.day_id < self.test_set_start_day_id) & df_all.sale_nbr.notnull() & df_all.buy_nbr.notnull()].copy()
        df_test = df_all.loc[(df_all.day_id >= self.test_set_start_day_id) & df_all.sale_nbr.notnull() & df_all.buy_nbr.notnull()].copy()


        df_edge_train = df_train.groupby(['sale_nbr', 'buy_nbr'], as_index=False).cnt.agg({'cnt_sum':'sum'})
        df_edge_train.drop('cnt_sum', axis=1, inplace=True)

        df_edge_test = df_test.groupby(['sale_nbr', 'buy_nbr'], as_index=False).cnt.agg({'cnt_sum':'sum'})
        df_edge_test.drop('cnt_sum', axis=1, inplace=True)

        ## 原有边中的共有边、原有边中消失的边、新边
        df_edge_common = pd.merge(df_edge_train, df_edge_test, how='inner', on=['sale_nbr', 'buy_nbr'])
        df_edge_perished = pd.concat([df_edge_train, df_edge_common]).drop_duplicates(keep=False)
        df_edge_new = pd.concat([df_edge_test, df_edge_common]).drop_duplicates(keep=False)

        df_test_gold = df_test.copy()
        df_test_gold.rename(columns={'cnt': 'cnt_gold', 'round': 'round_gold'}, inplace=True)

        print('读预测结果数据')
        df_test_prdt = pd.read_csv(filename_prdt, comment='#')



        ## eval
        ## 区分 原有共有边 和 新边
        df_test_gold_common = pd.merge(df_test_gold, df_edge_common, how='inner', on=['sale_nbr', 'buy_nbr'])
        df_test_gold_new = pd.merge(df_test_gold, df_edge_new, how='inner', on=['sale_nbr', 'buy_nbr'])


        ## 原有共有边 left join 预测结果
        df_result_valid = pd.merge(df_test_gold_common, df_test_prdt, how='left', on=['day_id', 'sale_nbr', 'buy_nbr'])

        # 剔除 == 0 的
        df_result_valid = df_result_valid.loc[(df_result_valid.cnt_gold != 0) & (df_result_valid.round_gold != 0)].copy()


        ## 额外的（新边 和 原有消失边 和 其他）
        df_test_prdt_extra = pd.concat([df_test_prdt, pd.merge(df_test_gold_common[['day_id', 'sale_nbr', 'buy_nbr']], df_test_prdt, how='inner', on=['day_id', 'sale_nbr', 'buy_nbr'])]).drop_duplicates(keep=False)

        df_result_extra = pd.merge(df_test_gold_new, df_test_prdt_extra, how='right', on=['day_id', 'sale_nbr', 'buy_nbr'])




        print('计算有效数据的MAE和MAPE')
        ## df_result_valid 指标
        len_valid_comm = len(df_result_valid.loc[df_result_valid.cnt.notnull()])

        df_result_valid['cnt_err'] = df_result_valid.cnt - df_result_valid.cnt_gold
        df_result_valid['round_err'] = df_result_valid['round'] - df_result_valid.round_gold

        cnt_mae_valid_comm = df_result_valid.cnt_err.abs().mean()
        round_mae_valid_comm = df_result_valid.round_err.abs().mean()

        cnt_mape_valid_comm = (df_result_valid.cnt_err / df_result_valid.cnt_gold).abs().mean()
        round_mape_valid_comm = (df_result_valid.round_err / df_result_valid.round_gold).abs().mean()


        print('计算用0填充后的MAE和MAPE')
        ## fillna(0)
        df_result_valid.cnt.fillna(0, inplace=True)
        df_result_valid['round'].fillna(0, inplace=True)

        len_valid_all = len(df_result_valid)

        df_result_valid['cnt_err'] = df_result_valid.cnt - df_result_valid.cnt_gold
        df_result_valid['round_err'] = df_result_valid['round'] - df_result_valid.round_gold

        cnt_mae_valid_all = df_result_valid.cnt_err.abs().mean()
        round_mae_valid_all = df_result_valid.round_err.abs().mean()

        cnt_mape_valid_all = (df_result_valid.cnt_err / df_result_valid.cnt_gold).abs().mean()
        round_mape_valid_all = (df_result_valid.round_err / df_result_valid.round_gold).abs().mean()


        print('计算额外的数据（不作要求），新边的MAE和MAPE')
        ## df_result_extra 新边 指标
        df_result_extra['cnt_err'] = df_result_extra.cnt - df_result_extra.cnt_gold
        df_result_extra['round_err'] = df_result_extra['round'] - df_result_extra.round_gold

        len_extra_new = len(df_result_extra.loc[df_result_extra.cnt_gold.notnull()])

        df_result_extra['cnt_err'] = df_result_extra.cnt - df_result_extra.cnt_gold
        df_result_extra['round_err'] = df_result_extra['round'] - df_result_extra.round_gold

        cnt_mae_extra_comm = df_result_extra.cnt_err.abs().mean()
        round_mae_extra_comm = df_result_extra.round_err.abs().mean()

        cnt_mape_extra_comm = (df_result_extra.cnt_err / df_result_extra.cnt_gold).abs().mean()
        round_mape_extra_comm = (df_result_extra.round_err / df_result_extra.round_gold).abs().mean()


        print('计算额外的数据（不作要求），消失边的正确记录数')
        ## 消失边 和 其他 指标
        len_extra_perished = len(df_result_extra.loc[df_result_extra.cnt_gold.isnull()])
        len_extra_perished_correct = len(df_result_extra.loc[df_result_extra.cnt_gold.isnull() & (df_result_extra.cnt <= 1) & (df_result_extra['round'] <= 1000)])


        print('保存结果文件')
        with open('./results/eval_{0}.txt'.format(int(datetime.datetime.now().timestamp())),  'w', encoding='utf-8') as f:
            evals = ''
            evals += '有效数据\n'
            evals += '记录数：{0}\n'.format(len_valid_comm)
            evals += 'cnt MAE：{0}\n'.format(cnt_mae_valid_comm)
            evals += 'round MAE：{0}\n'.format(round_mae_valid_comm)
            evals += 'cnt MAPE：{0}\n'.format(cnt_mape_valid_comm)
            evals += 'round MAPE：{0}\n'.format(round_mape_valid_comm)
            evals += '\n\n'

            evals += '用0填充后\n'
            evals += '记录数：{0}\n'.format(len_valid_all)
            evals += 'cnt MAE：{0}\n'.format(cnt_mae_valid_all)
            evals += 'round MAE：{0}\n'.format(round_mae_valid_all)
            evals += 'cnt MAPE：{0}\n'.format(cnt_mape_valid_all)
            evals += 'round MAPE：{0}\n'.format(round_mape_valid_all)
            evals += '\n\n'

            evals += '额外的数据（不作要求），新边\n'
            evals += '记录数：{0}\n'.format(len_extra_new)
            evals += 'cnt MAE：{0}\n'.format(cnt_mae_extra_comm)
            evals += 'round MAE：{0}\n'.format(round_mae_extra_comm)
            evals += 'cnt MAPE：{0}\n'.format(cnt_mape_extra_comm)
            evals += 'round MAPE：{0}\n'.format(round_mape_extra_comm)
            evals += '\n\n'

            evals += '额外的数据（不作要求），消失边\n'
            evals += '记录数：{0}\n'.format(len_extra_perished)
            evals += '正确的记录数：{0}\n'.format(len_extra_perished_correct)

            f.write(evals)

        return



def run():
    if not os.path.exists('./results'):
        os.mkdir('./results')

    with open('./setting.txt', 'r', encoding='utf-8') as f:
        parameters = f.read().strip('\n').split('\n')
        filename_all = parameters[0]
        filename_prdt = parameters[1]
        test_set_start_day_id = int(parameters[2])

    er = EvalResults(test_set_start_day_id=test_set_start_day_id)

    er.read_data_and_eval(filename_all=filename_all, filename_prdt=filename_prdt)

    print('done')
    input("Press Enter to exit...")


if __name__ == '__main__':

    run()
