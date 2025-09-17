import os
import csv
import json
import ast
import time
import logging
import argparse
from typing import Optional, List, Tuple

from pydantic import BaseModel
from google import genai
from google.genai import types

# ================== LOG ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============ MODELOS P/ RESPOSTA ============
class AvaliacaoCompetencia(BaseModel):
    competencia: str
    pontuacao: int
    justificativa: str

class SugestaoMelhora(BaseModel):
    trecho_original: str
    sugestao: str
    explicacao: str

class AnaliseRedacao(BaseModel):
    analise_geral: str
    pontos_fortes: List[str]
    sugestoes_de_melhora: List[SugestaoMelhora]
    avaliacoes_competencias: List[AvaliacaoCompetencia]
    nota_estimada: float

# ============== UTIL =================
def analise_para_json(analise: AnaliseRedacao | dict | None) -> str:
    try:
        if analise is None:
            return "{}"

        if isinstance(analise, dict):
            return json.dumps(analise, ensure_ascii=False)

        if hasattr(analise, "model_dump"):
            return json.dumps(analise.model_dump(exclude_none=True), ensure_ascii=False)

        if hasattr(analise, "json"):
            return analise.json(ensure_ascii=False)
        if hasattr(analise, "dict"):
            return json.dumps(analise.dict(), ensure_ascii=False)

        return json.dumps(analise, ensure_ascii=False, default=str)
    except Exception:
        logging.exception("Falha ao serializar a análise para JSON.")
        return "{}"


def parse_essay_field(essay_raw: str) -> str:
    """
    Campo `essay` vem como string de uma lista Python:
      "['parágrafo 1', 'parágrafo 2', ...]"
    Converte para texto com quebras de linha. Se não der, retorna o próprio texto.
    """
    if essay_raw is None:
        return ""
    essay_raw = essay_raw.strip()
    try:
        parsed = ast.literal_eval(essay_raw)
        if isinstance(parsed, list):
            return "\n\n".join(str(p).strip() for p in parsed)
        return str(parsed)
    except Exception:
        return essay_raw

def load_processed_pairs(out_csv: str) -> set[tuple[str, str]]:
    """
    Lê o CSV de saída (se existir) e retorna um conjunto de pares (tema, redacao)
    já processados, para evitar duplicidade e permitir retomar.
    """
    if not os.path.exists(out_csv):
        return set()
    processed = set()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tema = (row.get("tema") or "").strip()
            red = (row.get("redacao") or "").strip()
            if tema or red:
                processed.add((tema, red))
    logging.info("Registros já presentes no CSV de saída: %d", len(processed))
    return processed

def ensure_out_header(out_csv: str) -> None:
    """Garante cabeçalho no CSV de saída (apenas: tema, redacao, resultado_ia)."""
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tema", "redacao", "resultado_ia"],
        )
        if write_header:
            writer.writeheader()

def append_result(
    out_csv: str,
    tema: str,
    redacao_texto: str,
    resultado_json: str,
) -> None:
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tema", "redacao", "resultado_ia"],
        )
        writer.writerow(
            {
                "tema": tema,
                "redacao": redacao_texto,
                "resultado_ia": resultado_json,
            }
        )

# ============ CHAMADA À IA ============
def analisar_redacao_gemini(
    tema: str,
    texto_redacao: str,
    title: Optional[str] = None,
    competencia_original: Optional[List[int]] = None,
    score_original: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
) -> Tuple[Optional[AnaliseRedacao], Optional[str]]:
    """
    Envia a redação para avaliação do Gemini e retorna (analise, erro_str).
    """
    info_extra = []
    if title:
        info_extra.append(f"Título: {title}")
    if competencia_original:
        info_extra.append(f"Competências (rótulos humanos, 5x/200): {competencia_original}")
    if score_original:
        info_extra.append(f"Nota global (rótulo humano): {score_original}")

    cabecalho = "\n".join(info_extra)
    prompt = f"""Você é um corretor de redações especialista no ENEM.
Analise a redação abaixo e responda ESTRITAMENTE no schema JSON fornecido.

{cabecalho}

Tema: {tema}

Texto:
---
{texto_redacao}
---
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnaliseRedacao,
        temperature=temperature,
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            client = genai.Client()  # usa GEMINI_API_KEY do ambiente
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            analise = resp.parsed
            if not isinstance(analise, AnaliseRedacao):
                raise TypeError("A resposta não corresponde ao schema AnaliseRedacao.")
            return analise, None
        except Exception as e:
            last_err = str(e)
            logging.warning("Falha tentativa %d/%d: %s", attempt, max_retries, last_err)
            if attempt < max_retries:
                time.sleep(retry_backoff_s * attempt)

    return None, last_err or "Erro desconhecido"

# =============== PIPELINE ===============
def processar_csv(
    in_csv: str,
    out_csv: str,
    n: int,
    tema_padrao: str = "Tema não informado",
    pular_existentes: bool = True,
    mostrar_console: bool = False,
    offset: int = 0,
) -> None:
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("A variável de ambiente GEMINI_API_KEY não está configurada.")

    processed_pairs = load_processed_pairs(out_csv) if pular_existentes else set()
    ensure_out_header(out_csv)

    enviados = 0
    linha_idx = -1

    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            linha_idx += 1
            if linha_idx < offset:
                continue
            if enviados >= n:
                break

            title = (row.get("title") or "").strip()
            essay_raw = row.get("essay") or ""

            essay_text = parse_essay_field(essay_raw)
            tema = title if title else tema_padrao

            # Checagem de retomada com base em (tema, redacao)
            if pular_existentes and (tema.strip(), essay_text.strip()) in processed_pairs:
                logging.info("Pulando (tema, redacao) já presente no CSV de saída.")
                continue

            logging.info("Analisando título/tema='%s' ...", tema)

            analise, err = analisar_redacao_gemini(
                tema=tema,
                texto_redacao=essay_text,
                title=title,
                competencia_original=None,     # não precisamos mais desses campos no CSV
                score_original=None,
            )

            if analise:
                resultado_json = analise_para_json(analise)
            else:
                resultado_json = analise_para_json({"erro": err or "Falha na análise."})

            append_result(
                out_csv=out_csv,
                tema=tema,
                redacao_texto=essay_text,
                resultado_json=resultado_json,
            )

            if mostrar_console and analise:
                print("\n" + "=" * 60)
                print(f"TEMA: {tema}")
                print(f"NOTA ESTIMADA: {analise.nota_estimada:.1f}/1000.0")
                print("--- ANÁLISE GERAL ---")
                print(analise.analise_geral)

            enviados += 1

    logging.info("Concluído. Enviados %d itens ao Gemini e salvos em '%s'.", enviados, out_csv)

# =============== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lê um dataset CSV de redações e envia X primeiras ao Gemini, salvando resultados (tema, redacao, resultado_ia).")
    parser.add_argument("--in", dest="in_csv", required=True, help="Caminho do CSV de entrada (dataset).")
    parser.add_argument("--out", dest="out_csv", default="resultados_analise.csv", help="Caminho do CSV de saída.")
    parser.add_argument("--n", dest="n", type=int, default=100, help="Quantidade de redações a processar (X primeiras).")
    parser.add_argument("--offset", type=int, default=0, help="Pular as primeiras N linhas do arquivo de entrada antes de começar a contar.")
    parser.add_argument("--tema", dest="tema_padrao", default="Tema não informado", help="Tema padrão caso o título esteja vazio.")
    parser.add_argument("--nao-retomar", dest="retomar", action="store_false", help="Não pular registros já presentes no CSV de saída (baseado em tema+redacao).")
    parser.add_argument("--mostrar", action="store_true", help="Exibe um resumo da análise no console.")
    args = parser.parse_args()

    processar_csv(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        n=args.n,
        tema_padrao=args.tema_padrao,
        pular_existentes=args.retomar,
        mostrar_console=args.mostrar,
        offset=args.offset,
    )
