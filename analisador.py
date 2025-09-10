import os
import csv
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

from pydantic import BaseModel
from google import genai
from google.genai import types

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Esquema de resposta ---
@dataclass
class AvaliacaoCompetencia(BaseModel):
    competencia: str
    pontuacao: int
    justificativa: str

@dataclass
class SugestaoMelhora(BaseModel):
    trecho_original: str
    sugestao: str
    explicacao: str

@dataclass
class AnaliseRedacao(BaseModel):
    analise_geral: str
    pontos_fortes: list[str]
    sugestoes_de_melhora: list[SugestaoMelhora]
    avaliacoes_competencias: list[AvaliacaoCompetencia]
    nota_estimada: float

# --- Utilidades para CSV/Serialização ---
def analise_para_json(analise: AnaliseRedacao) -> str:
    """Serializa a resposta da IA para JSON (string), compatível com Pydantic v1/v2."""
    try:
        # Pydantic v2
        if hasattr(analise, "model_dump"):
            data = analise.model_dump(exclude_none=True)
            return json.dumps(data, ensure_ascii=False)

        # Pydantic v1
        if hasattr(analise, "dict"):
            data = analise.dict(exclude_none=True)
            return json.dumps(data, ensure_ascii=False)

        # Fallback: dataclass/objeto comum
        from dataclasses import asdict, is_dataclass
        if is_dataclass(analise):
            return json.dumps(asdict(analise), ensure_ascii=False, default=str)

        # Último recurso
        return json.dumps(analise, ensure_ascii=False, default=str)
    except Exception:
        logging.exception("Falha ao serializar a análise para JSON.")
        return "{}"


def salvar_em_csv(caminho_csv: str, tema: str, redacao: str, resultado_json: str) -> None:
    """Salva (ou anexa) uma linha no CSV com: tema, redacao, resultado_ia."""
    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)
    escrever_cabecalho = not os.path.exists(caminho_csv)

    with open(caminho_csv, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tema", "redacao", "resultado_ia"])
        if escrever_cabecalho:
            writer.writeheader()
        writer.writerow({
            "tema": tema,
            "redacao": redacao,
            "resultado_ia": resultado_json
        })
    logging.info("Linha registrada em %s", caminho_csv)

# --- Chamada à IA ---
def analisar_redacao(caminho_arquivo: str, tema_redacao: str) -> Tuple[Optional[AnaliseRedacao], Optional[str]]:
    """Lê a redação, envia ao Gemini e retorna (analise, texto_redacao)."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto_redacao = f.read()
        logging.info("Arquivo '%s' lido com sucesso.", caminho_arquivo)
    except FileNotFoundError:
        logging.error("Erro: Arquivo '%s' não encontrado.", caminho_arquivo)
        return None, None
    except Exception as e:
        logging.error("Erro ao ler arquivo: %s", e)
        return None, None

    prompt = f"""
Você é um corretor de redações experiente, especialista no modelo de avaliação do ENEM.
Analise a redação a seguir, cujo tema é '{tema_redacao}', e responda estritamente no schema JSON.

Texto da Redação:
---
{texto_redacao}
---
"""

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnaliseRedacao,
        temperature=0.2
    )

    try:
        logging.info("Enviando requisição para a API Gemini...")
        client = genai.Client()  # Usa GEMINI_API_KEY do ambiente
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=generation_config
        )
        analise = response.parsed
        if not isinstance(analise, AnaliseRedacao):
            raise TypeError("A resposta da API não corresponde ao schema esperado.")
        logging.info("Análise recebida com sucesso.")
        return analise, texto_redacao
    except Exception as e:
        logging.error("Erro na chamada à API ou processamento: %s", e)
        return None, texto_redacao  # Retorna texto para registrar mesmo em caso de falha

# --- Exibição opcional no console ---
def exibir_analise(analise: AnaliseRedacao):
    print("\n" + "="*50)
    print("      RESULTADO DA ANÁLISE DA REDAÇÃO")
    print("="*50 + "\n")
    print(f"NOTA ESTIMADA: {analise.nota_estimada:.1f} / 1000.0\n")
    print("--- ANÁLISE GERAL ---")
    print(analise.analise_geral + "\n")
    print("--- AVALIAÇÃO POR COMPETÊNCIA ---")
    for comp in analise.avaliacoes_competencias:
        print(f"\n* {comp.competencia}: {comp.pontuacao}/200")
        print(f"  Justificativa: {comp.justificativa}")
    print("\n--- PONTOS FORTES ---")
    for ponto in analise.pontos_fortes:
        print(f"- {ponto}")
    print("\n--- SUGESTÕES DE MELHORA ---")
    for i, s in enumerate(analise.sugestoes_de_melhora, 1):
        print(f"{i}. Trecho Original: \"{s.trecho_original}\"")
        print(f"   Sugestão: \"{s.sugestao}\"")
        print(f"   Explicação: {s.explicacao}\n")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa uma redação com a API Google GenAI e salva em CSV.")
    parser.add_argument("arquivo", type=str, help="Caminho para o arquivo .txt da redação.")
    parser.add_argument("-t", "--tema", type=str, required=True, help="Tema da redação.")
    parser.add_argument("-o", "--out", type=str, default="resultados_analise.csv",
                        help="Caminho do CSV de saída (default: resultados_analise.csv)")
    parser.add_argument("--mostrar", action="store_true",
                        help="Exibe a análise no console além de registrar no CSV.")
    args = parser.parse_args()

    if 'GEMINI_API_KEY' not in os.environ:
        logging.error("A variável de ambiente 'GEMINI_API_KEY' não está configurada.")
    else:
        analise, texto = analisar_redacao(args.arquivo, args.tema)
        # Serializa a análise (ou um objeto de erro) e salva no CSV
        if analise:
            resultado_json = analise_para_json(analise)
        else:
            # Guarda um JSON simples de erro, mas ainda registra tema e redação
            resultado_json = json.dumps(
                {"erro": "Falha na análise. Ver logs."},
                ensure_ascii=False
            )

        salvar_em_csv(args.out, args.tema, texto or "", resultado_json)

        if analise and args.mostrar:
            exibir_analise(analise)
