import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_student_info(df):
    df['correct_students'] = df.apply(lambda x: sum(x[:-1]), axis=1)
    df['total_students'] = df.apply(lambda x: len(x[:-1])-1, axis=1)
    return df
    
    
def get_qeustion_accuracy(df, drop_columns, new_column_name):
    df['question_acc'] = df.drop(drop_columns, axis=1).apply(lambda x: x.mean(), axis=1)
    return df
    
    
def get_student_accuracy(df, drop_columns, new_row_name, drop_acc_rate):
    column_list = list(df.columns)
    new_row = {}
    for drop_column in drop_columns:
        column_list.remove(drop_column)
        if drop_column=='question_id':
            new_row[drop_column] = new_row_name
        else:
            new_row[drop_column] = None

    drop_column_list = []
    for col in column_list:
        avg = df[col].values.mean()
        new_row[col] = avg
        if avg<drop_acc_rate:
            drop_column_list.append(col)

    df = pd.concat([df, pd.DataFrame(new_row, index=[0])])
    df = df[drop_columns+column_list]
    df = df.drop(drop_column_list, axis=1)
    
    df = df.reset_index(drop=True)
    return df


def get_metrics(save_info_df, llm_diff, student_diff):
    mae = mean_absolute_error(llm_diff, student_diff)
    mse = mean_squared_error(llm_diff, student_diff)
    rmse = np.sqrt(mse)

    save_info_df_ = pd.DataFrame({
        'what': ['mae', 'mse', 'rmse'],
        'who': ['both', 'both', 'both'],
        'score': [mae, mse, rmse]
    })
    save_info_df = pd.concat([save_info_df, save_info_df_])
    return save_info_df


def get_mean_vat_std(save_info_df, what, who, data_list):
    mean, var, std = np.mean(data_list), np.var(data_list), np.std(data_list)
    save_info_df_ = pd.DataFrame({
            "what": [what],
            "who": [who],
            "mean": [mean],
            "var": [var],
            "std": [std]
    })
    save_info_df = pd.concat([save_info_df, save_info_df_])
    return save_info_df


def save_info(llm_df, student_df, save_dir):
    question_diff_corr = stats.pearsonr(llm_df['question_diff'][:-1], student_df['question_diff'][:-1])
    question_acc_corr = stats.pearsonr(llm_df['question_acc'][:-1], student_df['question_acc'][:-1])
    
    save_info_df = pd.DataFrame({
        'what': ['question_diff', 'question_acc'],
        'who': ['both', 'both'],
        'correlation': [question_diff_corr[0], question_acc_corr[0]], 
        'p-value': [question_diff_corr[1], question_acc_corr[1]], 
    })
    
    save_info_df = get_metrics(save_info_df, llm_df['question_diff'][:-1], student_df['question_diff'][:-1])
    
    save_info_df = get_mean_vat_std(save_info_df, "question_acc", "llm", llm_df['question_acc'][:-1])
    save_info_df = get_mean_vat_std(save_info_df, "question_acc", "student", student_df['question_acc'][:-1])
    save_info_df = get_mean_vat_std(save_info_df, "student_acc", "llm", llm_df[llm_df['question_id']=='llm_acc'].values[0][5:])
    save_info_df = get_mean_vat_std(save_info_df, "student_acc", "student", student_df[student_df['question_id']=='student_acc'].values[0][5:])
    save_info_df = get_mean_vat_std(save_info_df, "question_diff", "llm", llm_df['question_diff'][:-1])
    save_info_df = get_mean_vat_std(save_info_df, "question_diff", "student", llm_df['question_diff'][:-1])

    save_info_df.to_csv(f'{save_dir}/info.csv', index=False)
    return save_info_df

    
def save_figure(llm_df, student_df, save_dir):
    ## 문제별 accuarcy histogram
    plt.figure()
    plt.hist(llm_df['question_acc'][:-1], bins=30, alpha=0.5, label="LLM Acc")
    plt.hist(student_df['question_acc'][:-1], bins=30, alpha=0.5, label="Student Acc")
    plt.legend(loc="upper left")
    plt.savefig(f'{save_dir}/quetsion_accuarcy_histogram.png')

    ## 학생별 accuarcy histogram
    plt.figure()
    plt.hist(llm_df[llm_df['question_id']=='llm_acc'].values[0][5:], bins=30, alpha=0.5, label="LLM Acc")
    plt.hist(student_df[student_df['question_id']=='student_acc'].values[0][5:], bins=30, alpha=0.5, label="Student Acc")
    plt.legend(loc="upper left")
    plt.savefig(f'{save_dir}/student_accuarcy_histogram.png')