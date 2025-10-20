import pandas as pd, json, re, unicodedata
from typing import Dict, Any, List

def strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def to_int_safe(x) -> int:
    try:
        if isinstance(x, (int,)):
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
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    s = str(value).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = json.loads(s)
            if isinstance(s2, str):
                return json.loads(s2)
        except Exception:
            pass
    return {}

def processar_arquivo(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, encoding="utf-8")
    pred_cols = [f"predicted_c{i}" for i in range(1, 6)] + ["predicted_total"]

    def compute_row(raw):
        payload = try_parse(raw)
        notas = extrair_notas(payload) if payload else {**{f"c{i}":0 for i in range(1,6)}, "total": 0}
        return [notas[f"c{i}"] for i in range(1,6)] + [notas["total"]]

    pred_values = df["resultado_ia"].apply(compute_row).tolist()
    pred_df = pd.DataFrame(pred_values, columns=pred_cols, index=df.index)
    out_df = pd.concat([df, pred_df], axis=1)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    processar_arquivo("essay-br-100-with-ia_.csv", "essay-br-100-with-ia_predicted_v2.csv")
