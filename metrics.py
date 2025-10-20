import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

def formatar_para_br(valor, decimais=2):
    """
    Formata um valor float para string usando a vírgula como separador decimal.
    Trata valores NaN retornando uma string vazia.
    """
    if pd.isna(valor):
        return ''

    formatacao_str = f"{round(valor, decimais):.{decimais}f}"

    return formatacao_str.replace('.', ',')

def calcular_metricas(df, score_columns):
    """
    Calcula MAE e QWK para todas as colunas de pontuação especificadas para um DataFrame/grupo.
    """
    results = {}

    results['num_essays'] = len(df)

    for comp_name, (real_col, pred_col) in score_columns.items():
        valid_data = df.dropna(subset=[real_col, pred_col])

        if valid_data.empty:
            results[f'MAE_{comp_name}'] = np.nan
            results[f'QWK_{comp_name}'] = np.nan
            continue

        y_true = valid_data[real_col].astype(int)
        y_pred = valid_data[pred_col].astype(int)

        mae = mean_absolute_error(y_true, y_pred)
        results[f'MAE_{comp_name}'] = mae

        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        results[f'QWK_{comp_name}'] = qwk

    return pd.Series(results)


def gerar_metricas_por_prompt(file_name_input, file_name_output):
    """
    Carrega o CSV, calcula as métricas por prompt e as métricas gerais,
    formata os números e salva o resultado final em um CSV separado por ponto e vírgula.
    """
    try:
        df = pd.read_csv(file_name_input)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_name_input}' não encontrado.")
        return

    score_columns = {
        'C1': ('real_c1', 'predicted_c1'),
        'C2': ('real_c2', 'predicted_c2'),
        'C3': ('real_c3', 'predicted_c3'),
        'C4': ('real_c4', 'predicted_c4'),
        'C5': ('real_c5', 'predicted_c5'),
        'total': ('score', 'predicted_total')
    }

    results_grouped_df = df.groupby('prompt').apply(
        lambda x: calcular_metricas(x, score_columns)
    ).reset_index()

    global_results = calcular_metricas(df, score_columns).to_dict()
    global_results['prompt'] = 'Geral'
    global_row_df = pd.DataFrame([global_results])

    mae_cols = [f'MAE_{comp}' for comp in score_columns.keys()]
    qwk_cols = [f'QWK_{comp}' for comp in score_columns.keys()]
    final_columns = ['prompt', 'num_essays'] + mae_cols + qwk_cols

    results_grouped_df['prompt'] = results_grouped_df['prompt'].astype(str)
    global_row_df = global_row_df[final_columns]

    final_results_df = pd.concat([results_grouped_df, global_row_df], ignore_index=True)

    final_results_df['num_essays'] = final_results_df['num_essays'].round(0).astype(int).astype(str)

    for col in mae_cols:
        final_results_df[col] = final_results_df[col].apply(lambda x: formatar_para_br(x, 2))

    for col in qwk_cols:
        final_results_df[col] = final_results_df[col].apply(lambda x: formatar_para_br(x, 4))

    final_results_df.to_csv(file_name_output, index=False, sep=';', encoding='utf-8')

    print(f"Métricas calculadas e salvas em: {file_name_output}")
    return final_results_df

input_file = 'essay-br-100-with-ia_predicted_v2_with_real.csv'
output_file = 'metricas.csv'

if __name__ == '__main__':
    gerar_metricas_por_prompt(input_file, output_file)
