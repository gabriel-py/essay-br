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
    Lê o CSV de saída (se existir) e retorna um conjunto de pares (prompt, essay)
    já processados, para evitar duplicidade e permitir retomar.
    """
    if not os.path.exists(out_csv):
        return set()
    processed = set()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = (row.get("prompt") or "").strip()
            essay = (row.get("essay") or "").strip()
            if prompt_id or essay:
                processed.add((prompt_id, essay))
    logging.info("Registros já presentes no CSV de saída: %d", len(processed))
    return processed


def ensure_out_header(out_csv: str) -> None:
    """
    Garante cabeçalho no CSV de saída:
    prompt, title, essay, competence, score, gemini_prompt, resultado_ia
    """
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "title",
                "essay",
                "competence",
                "score",
                "gemini_prompt",
                "resultado_ia",
            ],
        )
        if write_header:
            writer.writeheader()


def append_result(
    out_csv: str,
    prompt_id: str,
    title: str,
    essay_raw: str,
    competence: str,
    score: str,
    gemini_prompt: str,
    resultado_json: str,
) -> None:
    """
    Escreve no CSV de saída preservando os campos do CSV de entrada
    e acrescentando gemini_prompt e resultado_ia.
    """
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "title",
                "essay",
                "competence",
                "score",
                "gemini_prompt",
                "resultado_ia",
            ],
        )
        writer.writerow(
            {
                "prompt": prompt_id,
                "title": title,
                "essay": essay_raw,
                "competence": competence,
                "score": score,
                "gemini_prompt": gemini_prompt,
                "resultado_ia": resultado_json,
            }
        )


def build_prompt(
    base_prompt_text: str,
    tema: str,
    texto_redacao: str,
    title: Optional[str] = None,
    competence: Optional[str] = None,
    score: Optional[str] = None,
) -> str:
    """
    Monta o prompt final enviado ao Gemini:
    (texto do arquivo) + informações extras + tema + redação.
    """
    info_extra = []
    if title:
        info_extra.append(f"Título: {title}")
    if competence:
        info_extra.append(f"Competências (rótulos humanos): {competence}")
    if score:
        info_extra.append(f"Nota global (rótulo humano): {score}")

    cabecalho = "\n".join(info_extra).strip()
    cabecalho = f"\n{cabecalho}\n" if cabecalho else ""

    prompt = f"""{base_prompt_text.rstrip()}

{cabecalho}Tema: {tema}

Redação:
---
{texto_redacao}
---
"""
    return prompt


# ============ CHAMADA À IA ============
def analisar_redacao_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
) -> Tuple[Optional[AnaliseRedacao], Optional[str]]:
    """
    Envia o prompt para avaliação do Gemini e retorna (analise, erro_str).
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
    base_prompt_text: str,
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

            prompt_id = (row.get("prompt") or "").strip()
            title = (row.get("title") or "").strip()
            essay_raw = row.get("essay") or ""
            competence = (row.get("competence") or "").strip()
            score = (row.get("score") or "").strip()

            # Texto convertido para prompt (lista -> parágrafos)
            essay_text_for_prompt = parse_essay_field(essay_raw)
            tema = title if title else tema_padrao

            # Checagem de retomada com base em (prompt_id, essay_raw)
            key = (prompt_id.strip(), essay_raw.strip())
            if pular_existentes and key in processed_pairs:
                logging.info("Pulando (prompt, essay) já presente no CSV de saída: %s", prompt_id)
                continue

            logging.info("Analisando redação prompt_id='%s', título/tema='%s' ...", prompt_id, tema)

            # Monta o prompt final (arquivo + tema + redação)
            prompt_str = build_prompt(
                base_prompt_text=base_prompt_text,
                tema=tema,
                texto_redacao=essay_text_for_prompt,
                title=title,
                competence=competence,
                score=score,
            )

            analise, err = analisar_redacao_gemini(
                prompt=prompt_str,
            )

            if analise:
                resultado_json = analise_para_json(analise)
            else:
                resultado_json = analise_para_json({"erro": err or "Falha na análise."})

            # gemini_prompt = base de prompt usada (arquivo)
            # se quiser o prompt COMPLETO, troque para: gemini_prompt = prompt_str
            gemini_prompt = base_prompt_text

            append_result(
                out_csv=out_csv,
                prompt_id=prompt_id,
                title=title,
                essay_raw=essay_raw,  # mantém exatamente o que veio no CSV de entrada
                competence=competence,
                score=score,
                gemini_prompt=gemini_prompt,
                resultado_json=resultado_json,
            )

            if mostrar_console and analise:
                print("\n" + "=" * 60)
                print(f"PROMPT ID: {prompt_id}")
                print(f"TEMA: {tema}")
                print(f"NOTA ESTIMADA: {analise.nota_estimada:.1f}/1000.0")
                print("--- ANÁLISE GERAL ---")
                print(analise.analise_geral)

            enviados += 1

    logging.info("Concluído. Enviados %d itens ao Gemini e salvos em '%s'.", enviados, out_csv)


def carregar_prompt_base(prompt_file: str) -> str:
    if not prompt_file:
        raise ValueError("É necessário informar um arquivo de prompt base (ex: prompt1.txt).")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Arquivo de prompt não encontrado: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


# =============== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Lê um dataset CSV de redações (prompt, title, essay, competence, score) "
            "e envia X primeiras ao Gemini, salvando resultados em CSV com "
            "as mesmas colunas + gemini_prompt + resultado_ia."
        )
    )
    parser.add_argument("--in", default="essay-br.csv", dest="in_csv", help="Caminho do CSV de entrada (dataset).")
    parser.add_argument(
        "--out",
        dest="out_csv",
        default="essay-br-100-with-ia.csv",
        help="Caminho do CSV de saída (padrão: essay-br-100-with-ia.csv).",
    )
    parser.add_argument(
        "--prompt-file",
        dest="prompt_file",
        default="prompt1.txt",
        help="Arquivo de texto com o prompt base a ser enviado ao modelo (ex: prompt1.txt).",
    )
    parser.add_argument("--n", dest="n", type=int, default=100, help="Quantidade de redações a processar (X primeiras).")
    parser.add_argument("--offset", type=int, default=0, help="Pular as primeiras N linhas do arquivo de entrada antes de começar a contar.")
    parser.add_argument("--tema", dest="tema_padrao", default="Tema não informado", help="Tema padrão caso o título esteja vazio.")
    parser.add_argument(
        "--nao-retomar",
        dest="retomar",
        action="store_false",
        help="Não pular registros já presentes no CSV de saída (baseado em prompt+essay).",
    )
    parser.add_argument("--mostrar", action="store_true", help="Exibe um resumo da análise no console.")
    args = parser.parse_args()

    base_prompt_text = carregar_prompt_base(args.prompt_file)

    prompt_name, _ = os.path.splitext(os.path.basename(args.prompt_file))
    out_base, out_ext = os.path.splitext(args.out_csv)
    out_csv_final = f"{out_base}_{prompt_name}{out_ext}"

    processar_csv(
        in_csv=args.in_csv,
        out_csv=out_csv_final,
        n=args.n,
        base_prompt_text=base_prompt_text,
        tema_padrao=args.tema_padrao,
        pular_existentes=args.retomar,
        mostrar_console=args.mostrar,
        offset=args.offset,
    )
