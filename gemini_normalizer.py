import pandas as pd
import json
import re
import unicodedata
import ast
from typing import Dict, Any, List

def strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def to_int_safe(x) -> int:
    try:
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        s = str(x).strip().replace(",", ".")
        return int(float(s))
    except Exception:
        return 0

def infer_comp_index(nome_norm: str) -> int | None:
    m = re.search(r"competencia\s*[:(\-]*\s*([1-5])\b", nome_norm)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b([1-5])\b", nome_norm)
    if m2:
        return int(m2.group(1))
    n = nome_norm
    if ("norma" in n and ("padrao" in n or "lingua" in n)) or ("dominio" in n and "lingua" in n):
        return 1
    if "proposta" in n or "tema" in n or "conhecimento" in n:
        return 2
    if ("selecion" in n or "relacion" in n or "organ" in n) and ("argument" in n or "ponto de vista" in n or "opin" in n or "fatos" in n):
        return 3
    if "mecanism" in n or "coes" in n or "coer" in n or "progress" in n:
        return 4
    if "intervenc" in n or "direitos humanos" in n or "direitos" in n:
        return 5
    return None

def extrair_notas(payload: Dict[str, Any]) -> Dict[str, int]:
    """
    Extrai as notas previstas pela IA a partir do JSON em 'resultado_ia'.
    """
    notas = {f"c{i}": 0 for i in range(1, 6)}
    comps: List[Dict[str, Any]] = payload.get("avaliacoes_competencias", []) or []
    for item in comps:
        nome_raw = str(item.get("competencia", "") or "")
        nome_norm = strip_accents(nome_raw).lower()
        pts = to_int_safe(item.get("pontuacao", 0))
        idx = infer_comp_index(nome_norm)
        if idx and 1 <= idx <= 5:
            notas[f"c{idx}"] = pts
    notas["total"] = sum(notas[f"c{i}"] for i in range(1, 6))
    return notas

def try_parse(value: Any) -> Dict[str, Any]:
    """
    Tenta fazer o parse do campo 'resultado_ia' para dict.
    """
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    s = str(value).strip()
    # tentativa direta
    try:
        return json.loads(s)
    except Exception:
        pass
    # tentativa de JSON duplamente serializado
    try:
        s2 = json.loads(s)
        if isinstance(s2, str):
            return json.loads(s2)
    except Exception:
        pass
    return {}

def parse_real_competence(value: Any) -> List[int]:
    """
    Extrai as notas reais (humanas) do campo 'competence'.
    Espera algo como: [160, 160, 120, 120, 120]
    Retorna sempre uma lista de 5 ints: [c1, c2, c3, c4, c5]
    """
    if pd.isna(value):
        return [0, 0, 0, 0, 0]

    s = str(value).strip()
    if not s:
        return [0, 0, 0, 0, 0]

    arr = None

    # Tenta JSON direto
    try:
        parsed = json.loads(s)
        arr = parsed
    except Exception:
        # Tenta literal_eval (ex: "[160, 160, 120, 120, 120]")
        try:
            parsed = ast.literal_eval(s)
            arr = parsed
        except Exception:
            arr = None

    if isinstance(arr, dict):
        vals = []
        for i in range(1, 6):
            key1 = f"c{i}"
            key2 = f"C{i}"
            if key1 in arr:
                vals.append(to_int_safe(arr[key1]))
            elif key2 in arr:
                vals.append(to_int_safe(arr[key2]))
            else:
                vals.append(0)
        return vals

    if isinstance(arr, (list, tuple)):
        vals = [to_int_safe(v) for v in arr]
        if len(vals) >= 5:
            return vals[:5]
        else:
            vals = vals + [0] * (5 - len(vals))
            return vals

    return [0, 0, 0, 0, 0]

def processar_arquivo(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, encoding="utf-8")

    pred_cols = [f"predicted_c{i}" for i in range(1, 6)] + ["predicted_total"]

    def compute_row(raw):
        payload = try_parse(raw)
        notas = extrair_notas(payload) if payload else {**{f"c{i}": 0 for i in range(1, 6)}, "total": 0}
        return [notas[f"c{i}"] for i in range(1, 6)] + [notas["total"]]

    # gera as colunas predicted_*
    pred_values = df["resultado_ia"].apply(compute_row).tolist()
    pred_df = pd.DataFrame(pred_values, columns=pred_cols, index=df.index)

    # colunas reais (humanas) extraídas de 'competence'
    real_cols = [f"real_c{i}" for i in range(1, 6)]
    real_values = df["competence"].apply(parse_real_competence).tolist()
    real_df = pd.DataFrame(real_values, columns=real_cols, index=df.index)

    out_df = pd.concat([df, real_df, pred_df], axis=1)

    # calcula AE_score = |nota_prevista - nota_real|
    real_score_series = out_df["score"].apply(to_int_safe)
    out_df["AE_score"] = (out_df["predicted_total"] - real_score_series).abs()

    desired_order = [
        "prompt",
        "title",
        "essay",
        "competence",
        "real_c1",
        "real_c2",
        "real_c3",
        "real_c4",
        "real_c5",
        "score",
        "resultado_ia",
        "predicted_c1",
        "predicted_c2",
        "predicted_c3",
        "predicted_c4",
        "predicted_c5",
        "predicted_total",
        "AE_score",
    ]

    cols_present_ordered = [c for c in desired_order if c in out_df.columns]
    remaining_cols = [c for c in out_df.columns if c not in cols_present_ordered]
    final_cols = cols_present_ordered + remaining_cols

    out_df = out_df[final_cols]

    out_df.to_csv(output_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    input_file = "essay-br-100-with-ia_prompt3_cot"
    processar_arquivo("{}.csv".format(input_file), "{}_predicted_with_real.csv".format(input_file))
